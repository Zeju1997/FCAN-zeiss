from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import json
import os
import networks
import datasets
from train_utils import *
from eval import EvalMetrics

from PIL import Image

import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import copy

from torchvision.utils import save_image

from torch.autograd import Variable

from networks import DeepLab_ResNet101_MSC
from networks.ran_net import RAN
from networks.unet_layer import UNet_Layer
from networks.unet_encoder import UNet_encoder
from networks.aan_net import AAN

from hyperparameter_tuning import random_search


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data/resnet_pretrained"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet50"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def grey_to_rgb(img):
    img = torch.cat((img, img, img), 1)
    return img

def rescale_transform(img, out_range=(0, 1)):
    img = img - img.min()
    img /= (img.max() - img.min())
    img *= (out_range[1] - out_range[0])
    return img


activation = {}
def get_activation(name):
    def hook(model, input, output):
        # activation[name] = output.detach()
        activation[name] = output
    return hook

CIRRUS = "/home/zeju/Documents/zeiss_domain_adaption/splits/cirrus_samples.txt"
SPECTRALIS = "/home/zeju/Documents/zeiss_domain_adaption/splits/cirrus_samples.txt"
CIRRUS_SAMPLE = "/home/zeju/Documents/zeiss_domain_adaption/Retouch-dataset/pre_processed/Cirrus_part1/3c68f67cd2e2b41afa54bf6059f509d1/image/039.jpg"
SPECTRALIS_SAMPLE = "/home/zeju/Documents/zeiss_domain_adaption/Retouch-dataset/pre_processed/Spectralis_part1/7b2607e057592d507c4ec4732bae64c2/image/015.jpg"

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_F = []
        self.parameters_to_train_D = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.models["encoder"] = networks.ResnetEncoder(
        #     self.opt.num_layers, pretrained=False)
        # self.models["encoder"].to(self.device)
        # self.parameters_to_train += list(self.models["encoder"].parameters())
        #
        # self.models["decoder"] = networks.Decoder(
        #     self.models["encoder"].num_ch_enc)
        # self.models["decoder"].to(self.device)
        # self.parameters_to_train += list(self.models["decoder"].parameters())

        # Initialize the resnet50 and resnet101 model for this run
        model_50 = self.initialize_model("resnet50", requires_grad=False)
        self.models["resnet50"] = model_50
        self.models["resnet50"].to(self.device)

        model_101 = self.initialize_model("resnet101", requires_grad=False)
        self.models["resnet101"] = model_101
        self.models["resnet101"].to(self.device)

        # self.models["RAN"] = DeepLab_ResNet101_MSC(n_classes=21)
        self.models["RAN"] = RAN(in_channels=512, out_channels=21)
        self.models["RAN"].to(self.device)

        self.models["unet"] = networks.UNet(n_channels=1, n_classes=4)
        self.models["unet"].to(self.device)
        self.parameters_to_train += list(self.models["unet"].parameters())

        # Optimizers
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.load_model()

        model_unet_encoder = self.initialize_model("unet_encoder", requires_grad=False)
        self.models["unet_encoder"] = model_unet_encoder
        self.models["unet_encoder"].to(self.device)

        self.parameters_to_train_F += list(self.models["unet_encoder"].parameters())
        self.parameters_to_train_D += list(self.models["RAN"].parameters())

        self.optimizer_F = optim.Adam(self.parameters_to_train_F, self.opt.learning_rate)
        self.optimizer_D = optim.Adam(self.parameters_to_train_D, self.opt.learning_rate)

        self.models["unet_down4"] = UNet_Layer(output_layer='down4')
        self.models["unet_down4"].to(self.device)
        # self.parameters_to_train += list(self.models["unet"].parameters())
        # self.parameters_to_train += list(self.models["resnet50"].parameters())
        # self.parameters_to_train += list(self.models["resnet101"].parameters())


        '''
        w = Variable(torch.randn(3, 5), requires_grad=True)
        b = Variable(torch.randn(3, 5), requires_grad=True)
        self.parameters_to_train += w
        self.parameters_to_train += b
        '''



        '''
        self.model_optimizer = optim.Adam(self.parameters_to_train,
                                          self.opt.learning_rate)
        '''

        self.dataset = datasets.Retouch_dataset

        if self.opt.use_augmentation:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 #transforms.RandomRotation(degrees=(-20, 20)),
                                                 ])
        else:
            self.transform = None

        # self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.opt.ce_weighting).to(self.device),
        #                                      ignore_index=self.opt.ignore_idx)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.cirrus_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.vendor_dir,
            split='cirrus_samples',
            is_train=True,
            transform=self.transform)

        self.cirrus_loader = DataLoader(
            self.cirrus_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.spectralis_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.vendor_dir,
            split='spectralis_samples',
            is_train=True,
            transform=self.transform)

        self.spectralis_loader = DataLoader(
            self.spectralis_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.spectralis_aan_dataset_ = self.dataset(
            base_dir=self.opt.aan_dir,
            list_dir=self.opt.vendor_dir,
            split='spectralis_samples',
            is_train=True,
            transform=self.transform)

        self.spectralis_ann_loader = DataLoader(
            self.spectralis_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        train_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.list_dir,
            split='train',
            is_train=True,
            transform=self.transform)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        val_dataset = self.dataset(
            base_dir=self.opt.base_dir,
            list_dir=self.opt.list_dir,
            split='val',
            is_train=False,
            transform=self.transform)

        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_iter = iter(self.val_loader)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.num_total_steps_AAN = len(self.cirrus_dataset) + len(self.spectralis_dataset)

        self.writers = {}
        for mode in ["train", "val", "AAN"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    '''
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.opt.save_model:
                self.save_model()
    '''

    def train_aan(self, alpha=1e-14, w_sem=[1, 1, 1, 1, 1], w_sty=[1, 1, 1, 1, 1]):
        input = rescale_transform(torch.normal(mean=0.5, std=1, size=(1, 3, 512, 512), device=self.device))
        input.requires_grad_(True)

        '''
        source_img = Variable(grey_to_rgb(transforms.ToTensor()(pil_loader(CIRRUS_SAMPLE))).unsqueeze(0).cuda(), requires_grad=False)
        target_img = Variable(grey_to_rgb(transforms.ToTensor()(pil_loader(SPECTRALIS_SAMPLE))).unsqueeze(0).cuda(), requires_grad=False)
        save_image(source_img, 'source_img.png')
        save_image(target_img, 'target_img.png')
        '''

        '''
        inputs = {}
        inputs["source"] = source_img
        inputs["target"] = target_img
        L_RAN = self.compute_RAN_loss(inputs)
        '''
        self.alpha = alpha
        self.w_sem = w_sem
        self.w_sty = w_sty
        self.epoch = 0
        self.step = 0
        self.data = 0
        self.start_time = time.time()
        self.num_batch = 5
        self.num_sample = len(self.spectralis_loader)
        dataloader_iterator = iter(self.cirrus_loader)
        print("In total {} samples found.".format(self.num_sample))
        for idx, source in enumerate(self.spectralis_loader):
            vendor = source["case_name"][0].split(' ')[0]
            slice_name = source["case_name"][0].split(' ')[1]
            slice_idx = source["case_name"][0].split(' ')[2].zfill(3)
            out_dir = os.path.join(self.opt.vendor_dir, "aan_processed", "{}".format(vendor), "{}".format(slice_name), "image")
            out_path = os.path.join(out_dir, "{}.jpg".format(slice_idx))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if os.path.exists(out_path):
                # target = grey_to_rgb()
                for i in range(self.num_batch):
                    entity = next(dataloader_iterator)
                    if i == 0:
                        target = grey_to_rgb(entity["image"].to(self.device))
                    else:
                        target = torch.cat((target, grey_to_rgb(entity["image"].to(self.device))), dim=0)
                img = self.run_epoch_aan(input, grey_to_rgb(source["image"].to(self.device)), target)
                save_image(img, out_path)
                print("Image {}, in total {} samples".format(idx, self.num_sample))
                torch.cuda.empty_cache()

    def train_ran(self):
        input = rescale_transform(torch.normal(mean=0.5, std=1, size=(1, 3, 512, 512), device=self.device))
        input.requires_grad_(True)

        '''
        source_img = Variable(grey_to_rgb(transforms.ToTensor()(pil_loader(CIRRUS_SAMPLE))).unsqueeze(0).cuda(), requires_grad=False)
        target_img = Variable(grey_to_rgb(transforms.ToTensor()(pil_loader(SPECTRALIS_SAMPLE))).unsqueeze(0).cuda(), requires_grad=False)
        save_image(source_img, 'source_img.png')
        save_image(target_img, 'target_img.png')
        '''

        '''
        inputs = {}
        inputs["source"] = source_img
        inputs["target"] = target_img
        L_RAN = self.compute_RAN_loss(inputs)
        '''
        self.sample = 0
        self.num_sample = 0
        self.num_epoch = 10
        self.num_iteration = 10
        self.loss = nn.BCELoss()
        for self.epoch in range(self.num_epoch):

            # run adversarial branch
            self.run_epoch_adv()

            self.save_weight('unet_encoder')

            # run segmentation branch
            self.run_epoch_seg()

            self.save_weight('unet')

            sys.exit()

            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.opt.save_model:
                self.save_model()

            '''
            
            print(
                "[Epoch %d/%d] [Sample %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, self.num_epoch, self.sample, len(self.cirrus_loader), discriminator_loss.item(), generator_loss.item())
            )
            
            
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            '''
    def run_epoch_adv(self):
        for cirrus_sample, spectralis_sample in zip(self.cirrus_loader, self.spectralis_loader):   # spectralis aan processed data!!!!!!!!!!!!!
            '''
                F(x_t) = x: true data / spectralis / target
                F(x_s) = G(z): generated data / cirrus / source
                models["RAN"]: discriminator
                models["unet_encoder"]: feature generator / encoder
            '''
            '''
            dataloader_iterator = iter(self.cirrus_loader)
            for i in range(self.num_batch):
                entity = next(dataloader_iterator)
                if i == 0:
                    target = entity["image"].cuda()
                else:
                    target = torch.cat((target, entity["image"].cuda()), dim=0)
            inputs = {}
            inputs["source"] = source["image"].cuda()
            inputs["target"] = target
            inputs["label"] = source["label"]
            L_RAN = self.compute_RAN_loss(inputs)
            '''
            # zero the gradients of the feature extractor on each iteration
            self.optimizer_F.zero_grad()

            # generate feature encoding
            generated_data = self.models["unet_encoder"](cirrus_sample["image"].cuda()) # [1, 512, 32, 32]
            true_data = self.models["unet_encoder"](spectralis_sample["image"].cuda()) # [1, 512, 32, 32]

            true_labels = torch.ones_like(true_data) # [1, 512, 32, 32]
            false_labels = torch.zeros_like(true_data) # [1, 512, 32, 32]

            # Train the generator
            # We invert the labels here and don't train the discriminator because we want the generator
            # to make things the discriminator classifies as true
            generator_discriminator_out = self.models["RAN"](generated_data) # [1, 512, 32, 32]
            generator_loss = self.loss(generator_discriminator_out, true_labels)
            generator_loss.backward()
            self.optimizer_F.step()

            # Z = prediction_cirrus[1] * prediction_cirrus[2] * prediction_cirrus[3]

            # Train the discriminator on the true/generated data
            self.optimizer_D.zero_grad()
            true_discriminator_loss = self.models["RAN"](true_data)
            true_discriminator_loss = self.loss(true_discriminator_loss, true_labels)

            # add .detach() here think about this
            generator_discriminator_out = self.models["RAN"](generated_data.detach())
            generator_discriminator_loss = self.loss(generator_discriminator_out, false_labels)
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2

            discriminator_loss.backward()
            self.optimizer_D.step()

            self.sample += 1

            print(
                "[Epoch %d/%d] [Sample %d/%d] [D loss: %f] [G loss: %f]"
                % (self.epoch, self.num_epoch, self.sample, len(self.cirrus_loader), discriminator_loss.item(), generator_loss.item())
            )

    def run_epoch_seg(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            print("input", inputs)
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 #and self.step < 2000
            # late_phase = self.step % 2000 == 0
            '''
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()
            '''

            print(
                "[Epoch %d/%d] [Sample %d/%d] [D loss: %f]"
                % (self.epoch, self.num_epoch, batch_idx, len(self.train_loader), losses["loss"].item())
            )

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time
            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 #and self.step < 2000
            # late_phase = self.step % 2000 == 0
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1

    def run_epoch_aan(self, input, source, target):
        I = 500
        L_ANN = AAN()
        for self.step in range(I):
            before_op_time = time.time()

            if input.grad is not None:
                print("zero gradient")
                input.grad.zero_()

            # print("input", input)
            loss = L_ANN(input, source, target)
            loss.backward()

            # loss = torch.autograd.Variable(L_ANN.forward(), requires_grad=True)
            # grad = torch.autograd.grad(loss, input, allow_unused=True)

            w = 20000 * (I - self.step) / I
            # loss = self.compute_AAN_loss(input, source, target)
            # loss.backward()

            input = input.detach() - w * input.grad / torch.linalg.norm(input.grad.view(-1), ord=1)
            input.requires_grad_(True)

            # self.model_optimizer.zero_grad()
            # self.model_optimizer.step()

            duration = time.time() - before_op_time
            print("step", self.step, "loss", loss, "duration", duration)

            self.step += 1
        img = input.squeeze(0)
        return img

    def save_weight(self, model_save):
        if model_save == 'unet_encoder':
            print('save unet_encoder weights to unet...')
            model_load = 'unet'
        elif model_save == 'unet':
            print('save unet weights to unet_encoder...')
            model_load = 'unet_encoder'
        else:
            print('check model name.')

        model_dict = self.models[model_load].state_dict()
        pretrained_dict = self.models[model_save].state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.models[model_load].load_state_dict(model_dict)

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key != 'case_name':
                inputs[key] = ipt.to(self.device)

        outputs = {}

        # features = self.models["encoder"](inputs["image"])
        # preds = self.models["decoder"](features)
        preds = self.models["unet"](inputs["image"])

        outputs["pred"] = preds
        outputs["pred_idx"] = torch.argmax(preds, dim=1, keepdim=True)

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.compute_accuracy(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)

            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        pred = outputs['pred']
        target = inputs['label']
        # print(pred[0,,2,3])
        ce_loss = self.criterion(pred,
                                 target)
        mask = torch.zeros_like(ce_loss)

        mask_idx = (outputs["pred_idx"] > 0).squeeze(1)

        if mask_idx.sum() > 100:
            mask[mask_idx] = 1
        else:
            mask[(target > 0)] = 1

        outputs["mask"] = mask.unsqueeze(1)
        to_optimise = (ce_loss * mask).sum() / (mask.sum() + 1e-5)

        total_loss += to_optimise
        losses["loss"] = total_loss
        return losses

    def compute_AAN_loss(self, input, source, target):
        M_o = []
        M_s = []
        G_o = []
        G_t = []
        L = ['conv1', 'res2c', 'res3d', 'res4f', 'res5c']

        # model_ft.layer1[0].conv2.register_forward_hook(hook_fn)
        self.models["resnet50"].conv1.register_forward_hook(get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
        output = self.models["resnet50"](input)

        for layer in L:
            M_o.append(activation[layer])
            G_o.append(self.generate_style_image(activation[layer]))
            del activation[layer]

        self.models["resnet50"].conv1.register_forward_hook(get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
        output = self.models["resnet50"](source)

        for layer in L:
            M_s.append(activation[layer])
            # print("activation layer", layer, activation[layer])
            del activation[layer]

        for i in range(target.shape[0]):
            self.models["resnet50"].conv1.register_forward_hook(get_activation('conv1')) # conv1
            self.models["resnet50"].layer1[2].conv3.register_forward_hook(get_activation('res2c')) # res2c
            self.models["resnet50"].layer2[3].conv3.register_forward_hook(get_activation('res3d')) # res3d
            self.models["resnet50"].layer3[5].conv3.register_forward_hook(get_activation('res4f')) # res4f
            self.models["resnet50"].layer4[2].conv3.register_forward_hook(get_activation('res5c')) # res5c
            output = self.models["resnet50"](target[i, :, :, :].unsqueeze(0))
            for idx, layer in enumerate(L):
                if len(G_t) < 5:
                    G_t.append(self.generate_style_image(activation[layer]) / target.shape[0])
                else:
                    G_t[idx] += self.generate_style_image(activation[layer]) / target.shape[0]
                del activation[layer]

        alpha = 1e-14
        loss = torch.tensor(0, device=self.device)
        for i in range(len(M_o)):
            loss = loss + torch.dist(M_o[i], M_s[i], 2) + alpha * torch.dist(G_o[i], G_t[i], 2)

        # test_loss = Variable(loss, requires_grad=True)
        return loss


    def compute_RAN_loss(self, inputs):
        print("compute RAN loss ...")
        features = {}
        outputs = {}

        preds = self.models["unet"](inputs["source"])
        # print("preds shape", preds.shape)
        outputs["pred"] = preds
        outputs["pred_idx"] = torch.argmax(preds, dim=1, keepdim=True)
        target = inputs['label']
        L_seg = self.compute_losses(inputs, outputs)
        print("L_seg", L_seg['loss'])

        feature_map = self.models["unet_down4"](inputs["source"])
        features["source"] = self.models['RAN'](feature_map)
        # print("features source", features["source"])

        feature_map = self.models["unet_down4"](inputs["target"])
        features["target"] = self.models['RAN'](feature_map)
        # print("features target", features["target"])

        L_adv = - torch.mean(torch.log(features["target"])) - torch.mean(torch.log(1-features["source"]))
        # print("L_adv", L_adv)
        loss = torch.tensor(0, device=self.device)
        loss = L_adv - 5*L_seg['loss']
        print("L_RAN", loss)

        return loss

    def generate_style_image(self, inputs):
        c = inputs.shape[1]
        M_i = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2] * inputs.shape[3]).to(self.device)
        M_j = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2] * inputs.shape[3]).permute(0, 2, 1).to(self.device)
        style_image = torch.bmm(M_i, M_j)
        '''
        style_image1 = torch.zeros(c, c).to(self.device)
        for i in range(c):
            for j in range(c):
                a = torch.flatten(inputs[:, i, :, :])
                b = torch.flatten(inputs[:, j, :, :])
                style_image1[i, j] = torch.dot(a, b)
        test = style_image - style_image1
        print("style image sum", style_image.sum())
        print("test sum", test.sum())
        '''
        return style_image


    def compute_accuracy(self, inputs, outputs, losses):
        evaluation = EvalMetrics(outputs["pred_idx"],
                                 inputs['label'],
                                 n_classes=4)

        losses["accuracy/dice"] = evaluation.dice_coef()
        losses["accuracy/iou"] = evaluation.iou_coef()


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image("inputs/{}".format(j), inputs["image"][j].data, self.step)
            writer.add_image("labels/{}".format(j), normalize_image(inputs["label"][j].unsqueeze(0).data), self.step)
            writer.add_image("predictions/{}".format(j), normalize_image(outputs["pred_idx"][j].data), self.step)
            writer.add_image("positive_region/{}".format(j), outputs["mask"][j].data, self.step)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights...")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def set_parameter_requires_grad(self, model, requires_grad):
        if not requires_grad:
            for param in model.parameters():
                param.requires_grad = False


    def initialize_model(self, model_name, requires_grad):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None

        if model_name == "resnet50":
            """ Resnet50
            """
            print("Loading resnet50...")
            # model_ft = models.resnet50(pretrained=use_pretrained)
            model_ft = models.resnet50()
            model_ft.load_state_dict(torch.load('model/pretrained/resnet50.pth'))
            self.set_parameter_requires_grad(model_ft, requires_grad)
            # num_ftrs = model_ft.fc.in_features
            # model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        elif model_name == "resnet101":
            """ Resnet101
            """
            print("Loading resnet101...")
            # model_ft = models.resnet101(pretrained=use_pretrained)
            model_ft = models.resnet101()
            model_ft.load_state_dict(torch.load('model/pretrained/resnet101.pth'))
            self.set_parameter_requires_grad(model_ft, requires_grad)
            # num_ftrs = model_ft.fc.in_features
            # model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        elif model_name == "unet_encoder":
            """ Unet Encoder
            """
            print("Loading unet_encoder...")
            # model_ft = models.resnet101(pretrained=use_pretrained)

            # load part of the pre trained model
            pretrained_dict = self.models["unet"].state_dict()
            # print(self.models["unet"].state_dict())
            model_ft = UNet_encoder(n_channels=1, n_classes=4)
            model_dict = model_ft.state_dict()
            '''
            for k, v in pretrained_dict.items():
                print("k", k)
            print("num keys unet", len(pretrained_dict))
            for k, v in model_dict.items():
                print("l", k)
                print("l", v)
            print("num keys unet_encoder", len(model_dict))
            '''
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model_ft.load_state_dict(model_dict)
            '''
            for k, v in model_dict.items():
                print("l", k)
                print("l", v)
            '''
            self.set_parameter_requires_grad(model_ft, requires_grad)

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft



    def evaluate(self):
        self.dice_loss = 0
        self.iou_loss = 0
        self.vendor = "Spectralis"
        for idx, source in enumerate(self.spectralis_loader):
            outputs, losses = self.process_batch(source)
            self.compute_accuracy(source, outputs, losses)
            self.dice_loss += losses["accuracy/dice"] / len(self.cirrus_loader)
            self.iou_loss += losses["accuracy/iou"] / len(self.cirrus_loader)
        print("Evaluating {} data set, in total {} samples ...".format(self.vendor, len(self.spectralis_loader)))
        print("[Evaluation result] accuracy/dice: {}, accuracy/iou: {}".format(self.dice_loss, self.iou_loss))
        print("Evaluation finished!")


    def hyperparameter_tuning(self):
        input = rescale_transform(torch.normal(mean=0.5, std=1, size=(1, 1, 512, 512), device=self.device))
        input.requires_grad_(True)

        results = random_search(input, self.cirrus_loader, self.spectralis_loader,
                                                          random_search_spaces = {
                                                            # "lr": ([1e-3, 1e-4], 'log'),
                                                            # "lr_decay": ([0.8, 0.9], 'float'),
                                                            # "reg": ([1e-3, 1e-5], "log"), # [1e-4, 1e-6]
                                                            # "std": ([1e-2, 1e-5], "log"), # [1e-4, 1e-6]
                                                            # "hidden_size": ([150, 250], "int"),
                                                            # "num_layer": ([2, 4], "int"), # [2, 5]
                                                            "w_os_conv1": ([8, 10], 'float'),
                                                            "w_os_res2c": ([6, 8], 'float'),
                                                            "w_os_res3d": ([4, 6], 'float'),
                                                            "w_os_res4f": ([2, 4], 'float'),
                                                            "w_os_res5c": ([1, 3], 'float'),
                                                            "w_ot_conv1": ([8, 10], 'float'),
                                                            "w_ot_res2c": ([6, 8], 'float'),
                                                            "w_ot_res3d": ([4, 6], 'float'),
                                                            "w_ot_res4f": ([2, 4], 'float'),
                                                            "w_ot_res5c": ([1, 3], 'float'),
                                                            "alpha": ([5e-3, 5e-3], 'log'),
                                                           },
                                                            hyper_param = {
                                                            "lr_init": ([5000, 15000], 'float')
                                                           }, num_search=20, num_samples=4, epochs=1, patience=20, writer=self.writers["AAN"])
        # print("results", results)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
