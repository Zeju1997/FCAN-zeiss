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

'''
from exercise_code.solver import Solver
from exercise_code.networks.layer import Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.optimizer import SGD, Adam
from exercise_code.networks import (ClassificationNet, BCE,
                                    CrossEntropyFromLogits)
from trainer_resnet import train_aan
'''


ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']

def random_search(input, cirrus_loader, spectralis_loader,
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


    return findBestConfig(input, cirrus_loader, spectralis_loader, configs, hyper, epochs, patience, writer)


def findBestConfig(input, cirrus_loader, spectralis_loader, configs, hyper, EPOCHS, patience, writer):
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

    spectralis_iterator = iter(spectralis_loader)
    cirrus_iterator = iter(cirrus_loader)
    source_dict = next(spectralis_iterator)
    # source = grey_to_rgb(source_dict["image"].to(device))
    source = source_dict["image"].to(device)
    log_img(writer, "input_image", torch.cat((source, input), dim=0))
    num_batch = 5
    for i in range(num_batch):
        entity = next(cirrus_iterator)
        if i == 0:
            # target = grey_to_rgb(entity["image"].to(device))
            target = entity["image"].to(device)
        else:
            # target = torch.cat((target, grey_to_rgb(entity["image"].to(device))), dim=0)
            target = torch.cat((target, entity["image"].to(device)), dim=0)

    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format((i+1), len(configs)), configs[i], hyper[i])
        model = AAN(configs[i])
        # solver = Solver(model, train_loader, val_loader, **configs[i])
        # solver.train(epochs=EPOCHS, patience=PATIENCE)
        I = 2000
        iter_no_improve = 0
        input_i = input
        min_loss = 10000
        lr_init = hyper[i]["lr_init"]
        for step in range(I):
            before_op_time = time.time()
            if input.grad is not None:
                input.grad.zero_()
            loss = model(input_i, source, target)
            loss.backward()
            w = lr_init * (I - step) / I
            input_i = input_i.detach() - w * input_i.grad / torch.linalg.norm(input_i.grad.view(-1), ord=1)
            input_i.requires_grad_(True)
            duration = time.time() - before_op_time
            log_scalar(writer, "loss#{}".format(i), loss, step)
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
        img = input_i
        result = torch.cat((img, target), dim=0)
        log_img(writer, "result#{}".format(i), result)
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

def log_scalar(writer, name, loss, step):
        """Write an event to the tensorboard events file
        """
        writer.add_scalar("{}".format(name), loss, step)

def log_img(writer, name, img):
        """Write an event to the tensorboard events file
        """
        '''
        output_dir = os.path.join(os.getcwd(), "log", "segmentation", "AAN", "image")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "{}.png".format(name))
        '''
        for i in range(min(5, img.shape[0])):
            image = img[i, :, :, :]
            image[image < 0] = 0
            writer.add_image("{}/{}".format(name, i), image)
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
    img = torch.cat((img, img, img), 1)
    return img
