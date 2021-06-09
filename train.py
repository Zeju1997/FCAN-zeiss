from __future__ import absolute_import, division, print_function

from options import Options
import os
import argparse

from trainer_resnet import Trainer
# from trainer import Trainer

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)

options = Options()
opts = options.parse()

if __name__ == "__main__":
    # opts.ignore_idx = 0
    # opts.ce_weighting = [0.04, 0.32, 0.32, 0.32]
    # opts.learning_rate = 1e-3
    # Just for experiment

    # opts.list_dir = os.path.join(file_dir, "splits", "split_cirrus")
    # opts.use_augmentation = True
    # opts.save_model = True
    trainer = Trainer(opts)
    # trainer.normalize()

    trainer.train_aan()
    # trainer.train_coco()
    # trainer.train_ran()

    # trainer.evaluate()
    # trainer.hyperparameter_tuning()
