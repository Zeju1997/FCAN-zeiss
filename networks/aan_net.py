import sys

import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

activation = {}

class AAN(torch.nn.Module):

    def __init__(self, configs=None):
        super(AAN, self).__init__()
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the resnet50 and resnet101 model for this run
        model_50 = self.initialize_model("resnet50", requires_grad=False)
        self.models["resnet50"] = model_50
        self.models["resnet50"].to(self.device)
        self.w_os = {}
        self.w_ot = {}

        if configs is None:
            self.alpha = 2361759.1383429
            self.w_os["conv1"] = 1.4833282507245544
            self.w_os["res2c"] = 3.910816904760316
            self.w_os["res3d"] = 5.0634122235909444
            self.w_os["res4f"] = 7.435837022192873
            self.w_os["res5c"] = 7.2119672293035615
            self.w_ot["conv1"] = 1.515706416672807
            self.w_ot["res2c"] = 4.556745920981047
            self.w_ot["res3d"] = 7.846647656393519
            self.w_ot["res4f"] = 8.991458503687417
            self.w_ot["res5c"] = 19.40233534738993
        else:
            self.alpha = configs["alpha"]
            self.w_os["conv1"] = configs["w_os_conv1"]
            self.w_os["res2c"] = configs["w_os_res2c"]
            self.w_os["res3d"] = configs["w_os_res3d"]
            self.w_os["res4f"] = configs["w_os_res4f"]
            self.w_os["res5c"] = configs["w_os_res5c"]
            self.w_ot["conv1"] = configs["w_ot_conv1"]
            self.w_ot["res2c"] = configs["w_ot_res2c"]
            self.w_ot["res3d"] = configs["w_ot_res3d"]
            self.w_ot["res4f"] = configs["w_ot_res4f"]
            self.w_ot["res5c"] = configs["w_ot_res5c"]

    def forward(self, input, source, target):
        M_o = []
        M_s = []
        G_o = []
        G_t = []
        L = ['conv1', 'res2c', 'res3d', 'res4f', 'res5c']

        input = self.normalize_img(input)
        source = self.normalize_img(source)

        input.requires_grad_(True)

        # model_ft.layer1[0].conv2.register_forward_hook(hook_fn)
        self.models["resnet50"].conv1.register_forward_hook(self.get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(self.get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(self.get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(self.get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(self.get_activation('res5c')) # res5c
        output = self.models["resnet50"](input)

        for layer in L:
            M_o.append(activation[layer])
            # G_o.append(self.generate_style_image(activation[layer]))
            G_o.append(activation[layer])
            del activation[layer]

        self.models["resnet50"].conv1.register_forward_hook(self.get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(self.get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(self.get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(self.get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(self.get_activation('res5c')) # res5c
        output_source = self.models["resnet50"](source)

        for layer in L:
            M_s.append(activation[layer])
            # print("activation layer", layer, activation[layer])
            del activation[layer]

        for i in range(target.shape[0]):
            self.models["resnet50"].conv1.register_forward_hook(self.get_activation('conv1')) # conv1
            self.models["resnet50"].layer1[2].conv3.register_forward_hook(self.get_activation('res2c')) # res2c
            self.models["resnet50"].layer2[3].conv3.register_forward_hook(self.get_activation('res3d')) # res3d
            self.models["resnet50"].layer3[5].conv3.register_forward_hook(self.get_activation('res4f')) # res4f
            self.models["resnet50"].layer4[2].conv3.register_forward_hook(self.get_activation('res5c')) # res5c
            output_target = self.models["resnet50"](self.normalize_img(target[i, :, :, :].unsqueeze(0)))
            # output_target = self.models["resnet50"](target[i, :, :, :].unsqueeze(0))
            for idx, layer in enumerate(L):
                if len(G_t) < 5:
                    # G_t.append(self.generate_style_image(activation[layer]) / target.shape[0])
                    G_t.append(activation[layer] / target.shape[0])
                else:
                    # G_t[idx] += self.generate_style_image(activation[layer]) / target.shape[0]
                    G_t[idx] += activation[layer] / target.shape[0]
                del activation[layer]

        loss = torch.tensor(0, device=self.device)
        loss_seman = torch.tensor(0, device=self.device)
        loss_style = torch.tensor(0, device=self.device)
        loss_grey = torch.tensor(0, device=self.device)
        input_mean = torch.mean(input, 1)
        for i in range(len(M_o)):
            layer = L[i]
            loss = loss + self.w_os[layer] * self.content_loss(M_o[i], M_s[i]) + self.alpha * self.w_ot[layer] * self.style_loss(G_o[i], G_t[i])
            loss_seman = loss_seman + self.w_os[layer] * self.content_loss(M_o[i], M_s[i])
            loss_style = loss_style + self.alpha * self.w_ot[layer] * self.style_loss(G_o[i], G_t[i])
            # loss = loss + self.w_os[layer] * torch.dist(M_o[i], M_s[i], 2) + self.alpha * self.w_ot[layer] * torch.dist(G_o[i], G_t[i], 2) # + torch.mean(std_input) * 256
            # loss_seman = loss_seman + self.w_os[layer] * torch.dist(M_o[i], M_s[i], 2)
            # loss_style = loss_style + self.alpha * self.w_ot[layer] * torch.dist(G_o[i], G_t[i], 2)
        # loss = loss + torch.mean(torch.abs(input[:, 0, :, :] - input_mean) * 256) + torch.mean(torch.abs(input[:, 1, :, :] - input_mean) * 256) + torch.mean(torch.abs(input[:, 2, :, :] - input_mean) * 256)
        # loss_grey = torch.mean(torch.abs(input[:, 0, :, :] - input_mean) * 256) + torch.mean(torch.abs(input[:, 1, :, :] - input_mean) * 256) + torch.mean(torch.abs(input[:, 2, :, :] - input_mean) * 256)
        return loss, loss_seman, loss_style, loss_grey

    def normalize_img(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)
        return (img - mean) / std

    def style_loss(self, input, target):
        input_gram = self.gram_matrix(input)
        target_gram = self.gram_matrix(target)
        loss = F.mse_loss(input_gram, target_gram)
        return loss

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def content_loss(self, input, target):
        loss = F.mse_loss(input, target)
        return loss

    def get_activation(self, name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

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

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft
