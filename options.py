import os
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


class Options:
    def __init__(self):
        # TODO: Write the arguments
        self.parser = argparse.ArgumentParser(
            description="Retouch options")

        # PATHS
        self.parser.add_argument("--base_dir",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "Retouch-dataset/pre_processed"))

        self.parser.add_argument("--vendor_dir",
                                 type=str,
                                 help="path to the vendor",
                                 default=os.path.join(file_dir, "Retouch-dataset"))
        
        self.parser.add_argument("--list_dir",
                                 type=str,
                                 help="path to the split",
                                 default=os.path.join(file_dir, "splits", "split_cirrus"))

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "log"))

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the model",
                                 default='segmentation')
        # Architecture
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--ignore_idx",
                                 type=int,
                                 help="which class to be ignored for gradient",
                                 default=-1)

        # Optimization
        self.parser.add_argument("--use_augmentation",
                                 type=float,
                                 help="use data augmentation",
                                 default=True)

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)

        self.parser.add_argument("--ce_weighting",
                                 type=list,
                                 help="manual weighting",
                                 default=[0.25, 0.25, 0.25, 0.25])

        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=200)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=2)
        # System
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=100)

        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=15)

        self.parser.add_argument("--save_model",
                                 type=bool,
                                 help="save model every epoch",
                                 default=True)
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="folder of pretrain model",
                                 default=os.path.join(file_dir, "model/pretrained"))
        self.parser.add_argument("--models_to_load",
                                 type=list,
                                 help="pretrained model to load",
                                 default=['unet'])

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options