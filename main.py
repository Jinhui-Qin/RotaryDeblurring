import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.set_device(0)
from tensorboardX import SummaryWriter
from option import args, setup, cleanup
from data import Data
from model import Model
from loss import Loss
from optim import Optimizer
from train import Trainer


def main_worker(args):

    step_range = args.rotational_test_step

    save_dir = args.save_dir
    writer = SummaryWriter(os.path.join(save_dir, 'runs'))

    loaders = Data(args).get_loader()
    model = Model(args)

    model.parallelize()
    optimizer = Optimizer(args, model)
    criterion = Loss(args, model=model, optimizer=optimizer, writer=writer)
    trainer = Trainer(args, model, criterion, optimizer, loaders)

    for epoch in range(args.start_epoch, args.end_epoch+1):
        if args.do_train:
            trainer.train(epoch)

        if args.do_test:
            if epoch % args.test_every == 0:
                if trainer.epoch != epoch:
                    trainer.load(epoch)
                for step in step_range:
                    trainer.test(epoch, test_step=step)

        if args.rank == 0 or not args.launched:
            print('')

    trainer.imsaver.join_background()

    cleanup(args)

if __name__ == "__main__":
    main_worker(setup(args))