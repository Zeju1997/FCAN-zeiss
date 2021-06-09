import random
from math import log10
from itertools import product
import torch
from networks.aan_net import AAN
import time
import sys
from tensorboardX import SummaryWriter
import os
from torchvision.utils import save_image
import shutil
import json
import math

import torch.optim as optim

'''
from exercise_code.solver import Solver
from exercise_code.networks.layer import Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.optimizer import SGD, Adam
from exercise_code.networks import (ClassificationNet, BCE,
                                    CrossEntropyFromLogits)
from trainer_resnet import train_aan
'''


ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']

def random_search(source_loader, target_loader,
                        random_search_spaces = {
                            # "learning_rate": ([1e-3, 1e-4], 'log'),
                            # "lr_decay": ([0.8, 0.9], 'float'),
                            # "reg": ([1e-3, 1e-5], "log"), # [1e-4, 1e-6]
                            # "std": ([1e-2, 1e-5], "log"), # [1e-4, 1e-6]
                            # "hidden_size": ([150, 250], "int"),
                            # "num_layer": ([2, 4], "int"), # [2, 5]
                            "w_os_conv1": ([1e-1, 1], 'float'),
                            "w_os_res2c": ([1e-1, 1], 'float'),
                            "w_os_res3d": ([1e-1, 1], 'float'),
                            "w_os_res4f": ([1e-1, 1], 'float'),
                            "w_os_res5c": ([1e-1, 1], 'float'),
                            "w_ot_conv1": ([1e-1, 1], 'float'),
                            "w_ot_res2c": ([1e-1, 1], 'float'),
                            "w_ot_res3d": ([1e-1, 1], 'float'),
                            "w_ot_res4f": ([1e-1, 1], 'float'),
                            "w_ot_res5c": ([1e-1, 1], 'float'),
                            "alpha": ([1e-2, 1e-6], 'log'),
                      },
                      hyper_param = {
                            "lr_init": ([1e3, 1e5], 'log'),
                      }, num_search=20, num_samples=4, epochs=1, patience=5, writer=None):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.
    See the grid search documentation above.
    Additional/different optional arguments:
        - random_search_spaces: similar to grid search but values are of the
        form
        (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
        - num_search: number of times we sample in each int/float/log list
    """

    # writers = {}
    # writers["AAN"] = SummaryWriter(os.path.join(os.getcwd(), "log", "AAN"))

    configs = []
    hyper = []

    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))
        hyper.append(random_search_spaces_to_config(hyper_param))

    return findBestConfig(source_loader, target_loader, configs, hyper, epochs, patience, writer)

def normalize_img(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (img - mean) / std

def findBestConfig(source_loader, target_loader, configs, hyper, EPOCHS, patience, writer):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_val = None
    best_config = None
    best_model = None
    results = []

    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)
    source_dict = next(source_iterator)

    # source = grey_to_rgb(source_dict["image"].to(device))
    source = source_dict["image"].to(device)

    # input = rescale_transform(torch.normal(mean=0.5, std=1, size=source.shape, device=device))
    # input = torch.randn(source.data.size(), device=device)
    input = source
    input.requires_grad_(True)

    num_batch = 4
    for i in range(num_batch):
        entity_source = next(source_iterator)
        entity_target = next(target_iterator)
        if i == 0:
            # target = grey_to_rgb(entity_target["image"].to(device))
            # source_batch = grey_to_rgb(entity_source["image"].to(device))
            target = entity_target["image"].to(device)
            source_batch = entity_source["image"].to(device)
        else:
            # target = torch.cat((target, grey_to_rgb(entity_target["image"].to(device))), dim=0)
            # source_batch = torch.cat((source_batch, grey_to_rgb(entity_source["image"].to(device))), dim=0)
            target = torch.cat((target, entity_target["image"].to(device)), dim=0)
            source_batch = torch.cat((source_batch, entity_source["image"].to(device)), dim=0)

    log_img(writer, "input/source_images", source, source_batch)

    log_img_frequency = 50

    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format((i+1), len(configs)), configs[i], hyper[i])
        model = AAN(configs[i])
        # solver = Solver(model, train_loader, val_loader, **configs[i])
        # solver.train(epochs=EPOCHS, patience=PATIENCE)
        I = 400
        input_i = input
        min_loss = math.inf
        lr_init = hyper[i]["lr_init"]
        iter_no_improve = 0
        for step in range(I):
            before_op_time = time.time()
            if input.grad is not None:
                input.grad.zero_()
            loss, loss_seman, loss_style, loss_grey = model(input_i, source, target)
            loss.backward()
            w = lr_init * (I - step) / I
            input_i = input_i.detach() - w * input_i.grad / torch.linalg.norm(input_i.grad.view(-1), ord=1)
            input_i.requires_grad_(True)
            duration = time.time() - before_op_time
            log_scalar(writer, "loss#{}".format(i), loss, loss_seman, loss_style, loss_grey, step)
            print("step", step, "loss", loss, "duration", duration)
            # early stopping
            if min_loss > loss.item():
                iter_no_improve = 0
                min_loss = loss.item()
            else:
                iter_no_improve += 1
                # Check early stopping condition
                if iter_no_improve == patience:
                    print('Early stopping!')
                    break
            if step % log_img_frequency == 0:
                img = input_i
                log_img(writer, "result#{}/target_images".format(i), img, target)
        log_text(writer, "config#{}".format(i), hyper[i])
        log_text(writer, "config#{}".format(i), configs[i])
        results.append(configs[i])

        '''
        loss = model(input, grey_to_rgb(source["image"].to(device)), target)
        results.append(solver.best_model_stats)

        if not best_val or solver.best_model_stats["val_loss"] < best_val:
            best_val, best_model,\
            best_config = solver.best_model_stats["val_loss"], model, configs[i]
        '''
    return list(results)

def log_scalar(writer, name, loss, loss_seman, loss_style, loss_grey, step):
        """Write an event to the tensorboard events file
        """
        loss_dict = {
            # "grey_loss": loss_grey,
            "semantic_loss": loss_seman,
            "style_loss": loss_style,
            "total_loss": loss,
        }
        for k, v in loss_dict.items():
            writer.add_scalar("{}/{}".format(name, k), v, step)

def log_img(writer, name, img, img_batch):
        """Write an event to the tensorboard events file
        """
        '''
        output_dir = os.path.join(os.getcwd(), "log", "segmentation", "AAN", "image")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "{}.png".format(name))
        '''
        img.data.clamp_(0, 1)
        img_batch.data.clamp_(0, 1)
        image = img[0, :, :, :]
        writer.add_image("{}/{}".format(name, 0), image)
        for i in range(min(3, img_batch.shape[0])):
            image_batch = img_batch[i, :, :, :]
            image_batch[image_batch < 0] = 0
            writer.add_image("{}/{}".format(name, i+1), image_batch)
        # save_image(img[0, :, :, :], output_path)

def log_text(writer, name, config):
        """Write an event to the tensorboard events file
        """
        result = json.dumps(config)
        writer.add_text("{}".format(name), result)
        '''
        for idx, (k, v) in enumerate(config.items()):
            print("key", k)
            if idx < 5:
                writer.add_text("{}".format(name), "{}:{}".format(k, v))
            elif 4 < idx < 10:
                writer.add_text("{}".format(name), "{}:{}".format(k, v))
            else:
        '''




def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """
    
    config = {}

    for key, (rng, mode)  in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <=0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10**(sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config


def grey_to_rgb(img):
    if img.shape[1] == 1:
        img = torch.cat((img, img, img), 1)
    return img


def rescale_transform(img, out_range=(0, 1)):
    img = img - img.min()
    img /= (img.max() - img.min())
    img *= (out_range[1] - out_range[0])
    return img
