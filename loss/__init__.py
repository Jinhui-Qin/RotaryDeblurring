import os
from importlib import import_module
import torch
from torch import nn
import torch.distributed as dist
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # https://github.com/matplotlib/matplotlib/issues/3466
from .metric import PSNR, SSIM, ssim_loss

class Loss(torch.nn.modules.loss._Loss):
    def __init__(self, args, epoch=None, model=None, optimizer=None, writer=None):
        super(Loss, self).__init__()

        self.args = args
        self.save_dir = args.save_dir

        self.rotational = args.rotational
        self.testangle = 3.0
        self.step_range = args.rotational_test_step
        self.writer = writer

        self.rgb_range = args.rgb_range
        self.device_type = args.device_type
        self.synchronized = False

        self.epoch = args.start_epoch if epoch is None else epoch
        self.save_name = os.path.join(self.save_dir, 'loss.pt')

        # self.training = True
        self.validating = False
        self.testing = False
        self.mode = 'train'
        self.modes = ('train', 'val', 'test')

        # Loss
        self.loss = nn.ModuleDict()
        self.loss_types = []
        self.weight = {}

        self.loss_stat = {mode:{} for mode in self.modes}
        # loss_stat[mode][loss_type][epoch] = loss_value
        # loss_stat[mode]['Total'][epoch] = loss_total

        for weighted_loss in args.loss.split('+'):
            w, l = weighted_loss.split('*')
            l = l.upper()
            if l in ('ABS', 'L1'):
                loss_type = 'L1'
                func = nn.L1Loss()
            elif l in ('MSE', 'L2'):
                loss_type = 'L2'
                func = nn.MSELoss()
            elif l in ('SSIM'):
                loss_type = 'SSIM'
                func = ssim_loss(device_type=args.device_type)
            elif l in ('ADV', 'GAN'):
                loss_type = 'ADV'
                m = import_module('loss.adversarial')
                func = getattr(m, 'Adversarial')(args, model, optimizer)
            else:
                loss_type = l
                m = import_module*'loss.{}'.format(l.lower())
                func = getattr(m, l)(args)

            self.loss_types += [loss_type]
            self.loss[loss_type] = func
            self.weight[loss_type] = float(w)

        print('Loss function: {}'.format(args.loss))

        # Metrics
        self.do_measure = args.metric.lower() != 'none'

        self.metric = nn.ModuleDict()
        self.metric_types = []
        self.metric_stat = {mode:{} for mode in self.modes}
        # metric_stat[mode][metric_type][epoch] = metric_value

        if self.do_measure:
            for metric_type in args.metric.split(','):
                metric_type = metric_type.upper()
                if metric_type == 'PSNR':
                    metric_func = PSNR()
                elif metric_type == 'SSIM':
                    metric_func = SSIM(args.device_type)    # single precision
                else:
                    raise NotImplementedError

                self.metric_types += [metric_type]
                self.metric[metric_type] = metric_func

        print('Metrics: {}'.format(args.metric))
        self.metric_previous = 0.0

        if args.start_epoch != 1:
            self.load(args.start_epoch - 1)

        for mode in self.modes:
            for loss_type in self.loss:
                if loss_type not in self.loss_stat[mode]:
                    self.loss_stat[mode][loss_type] = {}   # initialize loss

            if 'Total' not in self.loss_stat[mode]:
                self.loss_stat[mode]['Total'] = {}

            if self.do_measure:
                for metric_type in self.metric:
                    if metric_type not in self.metric_stat[mode]:
                        self.metric_stat[mode][metric_type] = {}

        self.count = 0
        self.count_m = 0

        self.to(args.device, dtype=args.dtype)

    def set_teststep(self, teststep=70):
        self.testangle = teststep

    def train(self, mode=True):
        super(Loss, self).train(mode)
        if mode:
            self.validating = False
            self.testing = False
            self.mode = 'train'
        else:   # default test mode
            self.validating = False
            self.testing = True
            self.mode = 'test'

    def validate(self):
        super(Loss, self).eval()
        # self.training = False
        self.validating = True
        self.testing = False
        self.mode = 'val'

    def test(self):
        super(Loss, self).eval()
        # self.training = False
        self.validating = False
        self.testing = True
        self.mode = 'test'

    def forward(self, input, target):
        self.synchronized = False
        loss = 0

        def _ms_forward(input, target, func):
            if isinstance(input, (list, tuple)): # loss for list output
                _loss = []
                for (input_i, target_i) in zip(input, target):
                    _loss += [func(input_i, target_i)]
                return sum(_loss)
            elif isinstance(input, dict):   # loss for dict output
                _loss = []
                for key in input:
                    _loss += [func(input[key], target[key])]
                return sum(_loss)
            else:   # loss for tensor output
                return func(input, target)

        # initialize
        if self.count == 0:
            for loss_type in self.loss_types:
                if self.rotational and (self.mode == 'test'):
                    try:
                        self.loss_stat[self.mode][loss_type][self.epoch][self.testangle] = 0
                    except:
                        self.loss_stat[self.mode][loss_type][self.epoch] = {}
                        self.loss_stat[self.mode][loss_type][self.epoch][self.testangle] = 0
                else:
                    self.loss_stat[self.mode][loss_type][self.epoch] = 0

            if self.rotational and (self.mode == 'test'):
                try:
                    self.loss_stat[self.mode]['Total'][self.epoch][self.testangle] = 0
                except:
                    self.loss_stat[self.mode]['Total'][self.epoch] = {}
                    self.loss_stat[self.mode]['Total'][self.epoch][self.testangle] = 0
            else:
                self.loss_stat[self.mode]['Total'][self.epoch] = 0

        if isinstance(input, list):
            count = input[0].shape[0]
        else:   # Tensor
            count = input.shape[0]  # batch size

        isnan = False
        for loss_type in self.loss_types:
            if loss_type == 'ADV':
                _loss = self.loss[loss_type](input[0], target[0], self.training) * self.weight[loss_type]
            else:
                _loss = _ms_forward(input, target, self.loss[loss_type]) * self.weight[loss_type]

            if torch.isnan(_loss):
                isnan = True    # skip recording (will also be skipped at backprop)
            else:
                if self.rotational and (self.mode == 'test'):
                    self.loss_stat[self.mode][loss_type][self.epoch][self.testangle] += _loss.item() * count
                    self.loss_stat[self.mode]['Total'][self.epoch][self.testangle] += _loss.item() * count
                else:
                    self.loss_stat[self.mode][loss_type][self.epoch] += _loss.item() * count
                    self.loss_stat[self.mode]['Total'][self.epoch] += _loss.item() * count

            loss += _loss

        if not isnan:
            self.count += count

        if not self.training and self.do_measure:
            self.measure(input, target)

        return loss

    def measure(self, input, target, metric_rot=False):
        if isinstance(input, (list, tuple)):
            self.measure(input[0], target[0], metric_rot=metric_rot)
            return
        elif isinstance(input, dict):
            first_key = list(input.keys())[0]
            self.measure(input[first_key], target[first_key])
            return
        else:
            pass

        if self.count_m == 0:
            for metric_type in self.metric_stat[self.mode]:
                if self.rotational and (self.mode == 'test'):
                    try:
                        self.metric_stat[self.mode][metric_type][self.epoch][self.testangle] = 0
                    except:
                        self.metric_stat[self.mode][metric_type][self.epoch] = {}
                        self.metric_stat[self.mode][metric_type][self.epoch][self.testangle] = 0
                else:
                    self.metric_stat[self.mode][metric_type][self.epoch] = 0

        if isinstance(input, list):
            count = input[0].shape[0]
        else:   # Tensor
            count = input.shape[0]  # batch size

        _metric = None
        for metric_type in self.metric_stat[self.mode]:
            input = input.clamp(0, self.rgb_range)  # not in_place
            if self.rgb_range == 255:
                input.round_()
            _metric = self.metric[metric_type](input, target)
            if self.rotational and (self.mode == 'test'):
                if _metric != None:
                    self.metric_stat[self.mode][metric_type][self.epoch][self.testangle] += _metric.item() * count
                else:
                    continue
            else:
                self.metric_stat[self.mode][metric_type][self.epoch] += _metric.item() * count

        self.count_m += count

        return

    def normalize(self):
        if self.args.distributed:
            dist.barrier()
            if not self.synchronized:
                self.all_reduce()

        if self.count > 0:
            for loss_type in self.loss_stat[self.mode]: # including 'Total'
                if self.rotational and (self.mode == 'test'):
                    self.loss_stat[self.mode][loss_type][self.epoch][self.testangle] /= self.count
                else:
                    self.loss_stat[self.mode][loss_type][self.epoch] /= self.count
            self.count = 0

        if self.count_m > 0:
            for metric_type in self.metric_stat[self.mode]:
                if self.rotational and (self.mode == 'test'):
                    self.metric_stat[self.mode][metric_type][self.epoch][self.testangle] /= self.count_m
                else:
                    self.metric_stat[self.mode][metric_type][self.epoch] /= self.count_m
            self.count_m = 0

        return

    def all_reduce(self, epoch=None):
        # synchronize loss for distributed GPU processes

        if epoch is None:
            epoch = self.epoch

        def _reduce_value(value, ReduceOp=dist.ReduceOp.SUM):
            value_tensor = torch.Tensor([value]).to(self.args.device, self.args.dtype, non_blocking=True)
            dist.all_reduce(value_tensor, ReduceOp, async_op=False)
            value = value_tensor.item()
            del value_tensor

            return value

        dist.barrier()
        if self.count > 0:  # I assume this should be true
            self.count = _reduce_value(self.count, dist.ReduceOp.SUM)

            for loss_type in self.loss_stat[self.mode]:
                if self.rotational and (self.mode == 'test'):
                    self.loss_stat[self.mode][loss_type][epoch][self.testangle] = _reduce_value(
                        self.loss_stat[self.mode][loss_type][epoch][self.testangle],
                        dist.ReduceOp.SUM
                    )
                else:
                    self.loss_stat[self.mode][loss_type][epoch] = _reduce_value(
                        self.loss_stat[self.mode][loss_type][epoch],
                        dist.ReduceOp.SUM
                    )

        if self.count_m > 0:
            self.count_m = _reduce_value(self.count_m, dist.ReduceOp.SUM)

            if self.rotational and (self.mode == 'test'):
                for metric_type in self.metric_stat[self.mode]:
                    self.metric_stat[self.mode][metric_type][epoch][self.testangle] = _reduce_value(
                        self.metric_stat[self.mode][metric_type][epoch],
                        dist.ReduceOp.SUM
                    )
            else:
                for metric_type in self.metric_stat[self.mode]:
                    self.metric_stat[self.mode][metric_type][epoch] = _reduce_value(
                        self.metric_stat[self.mode][metric_type][epoch],
                        dist.ReduceOp.SUM
                    )

        self.synchronized = True

        return

    def print_metrics(self):

        print(self.get_metric_desc())
        return

    def get_last_loss(self):
        if self.rotational and (self.mode == 'test'):
            return self.loss_stat[self.mode]['Total'][self.epoch][self.testangle]
        else:
            return self.loss_stat[self.mode]['Total'][self.epoch]

    def get_loss_desc(self):

        if self.mode == 'train':
            desc_prefix = 'Train'
        elif self.mode == 'val':
            desc_prefix = 'Validation'
        else:
            desc_prefix = 'Test'

        if self.rotational and (self.mode == 'test'):
            loss = self.loss_stat[self.mode]['Total'][self.epoch][self.testangle]
        else:
            loss = self.loss_stat[self.mode]['Total'][self.epoch]

        if self.count > 0:
            loss /= self.count
        desc = '{} Loss: {:.1f}'.format(desc_prefix, loss)

        if self.mode in ('val', 'test'):
            metric_desc = self.get_metric_desc()
            desc = '{}{}'.format(desc, metric_desc)

        return desc

    def get_metric_desc(self):
        desc = ''
        for metric_type in self.metric_stat[self.mode]:
            if self.rotational and (self.mode == 'test'):
                measured = self.metric_stat[self.mode][metric_type][self.epoch][self.testangle]
            else:
                measured = self.metric_stat[self.mode][metric_type][self.epoch]

            if self.count_m > 0:
                measured /= self.count_m

            if metric_type == 'PSNR':
                desc += ' {}: {:2.2f}'.format(metric_type, measured)
            elif metric_type == 'SSIM':
                desc += ' {}: {:1.4f}'.format(metric_type, measured)
            else:
                desc += ' {}: {:2.4f}'.format(metric_type, measured)

        return desc

    def step(self, plot_name=None):
        self.normalize()
        self.plot(plot_name)
        if not self.training and self.do_measure:
            if self.rotational and self.mode == 'test':
                self.plot_metric_summary()

        return

    def save(self):

        state = {
            'loss_stat': self.loss_stat,
            'metric_stat': self.metric_stat,
        }
        torch.save(state, self.save_name)

        return

    def load(self, epoch=None):

        print('Loading loss record from {}'.format(self.save_name))
        if os.path.exists(self.save_name):
            state = torch.load(self.save_name, map_location=self.args.device)

            self.loss_stat = state['loss_stat']
            if 'metric_stat' in state:
                self.metric_stat = state['metric_stat']
            else:
                pass
        else:
            print('no loss record found for {}!'.format(self.save_name))

        if epoch is not None:
            self.epoch = epoch

        return

    def plot(self, metric=False):

        if self.rotational:
            self.plot_loss_rotation_summary()
            if metric and self.mode == 'test':
                self.plot_metric_summary()
        return

    def plot_loss_rotation_summary(self):

        if self.mode == 'test':
            # self.loss_stat[self.mode][loss_type][self.epoch][self.testangle]
            loss_dic = {}
            for loss_type, loss_record in self.loss_stat[self.mode].items():
                loss_dic[loss_type] = {}
                for angle in self.step_range:
                    key = '{}_theta_{}'.format(self.mode, angle)
                    try:
                        loss_dic[loss_type][key] = loss_record[self.epoch][angle]
                    except:
                        continue

            for loss_type in self.loss_stat[self.mode].keys():
                tag = os.path.join(self.mode, 'Loss_' + loss_type)
                self.writer.add_scalars(tag, loss_dic[loss_type], global_step=self.epoch)

        elif self.mode == 'train':
            # self.loss_stat[self.mode][loss_type][self.epoch]
            for loss_type, loss_record in self.loss_stat[self.mode].items():
                tag = os.path.join(self.mode, loss_type)
                self.writer.add_scalar(tag, loss_record[self.epoch], global_step=self.epoch)

        else:
            print('{} in plot_loss_rotation_summary() is not implemented'.format(self.mode))

        return

    def plot_metric_summary(self):
        # self.metric_stat[self.mode][metric_type][self.epoch][self.testangle]

        metric_dic = {}
        for metric_type, metric_record in self.metric_stat[self.mode].items():
            metric_dic[metric_type] = {}
            for angle in self.step_range:
                key = '{}_theta_{}'.format(self.mode, angle)
                try:
                    metric_dic[metric_type][key] = metric_record[self.epoch][angle]
                except:
                    continue

        for metric_type in self.metric_stat[self.mode].keys():
            tag = os.path.join(self.mode, 'Metric_' + metric_type)
            self.writer.add_scalars(tag, metric_dic[metric_type], global_step=self.epoch)

        return

    def sort(self):
        # sort the loss/metric record
        for mode in self.modes:
            for loss_type, loss_epochs in self.loss_stat[mode].items():
                self.loss_stat[mode][loss_type] = {epoch: loss_epochs[epoch] for epoch in sorted(loss_epochs)}

            for metric_type, metric_epochs in self.metric_stat[mode].items():
                self.metric_stat[mode][metric_type] = {epoch: metric_epochs[epoch] for epoch in sorted(metric_epochs)}

        return self