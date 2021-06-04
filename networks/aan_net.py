import sys

import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models

activation = {}

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=4):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.__class__ = models.resnet.ResNet

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
            print("right config")
            self.alpha = 3.4400135531611185e-05
            self.w_os["conv1"] = 0.12467511325675217
            self.w_os["res2c"] = 0.841792665975288
            self.w_os["res3d"] = 0.7283800534768836
            self.w_os["res4f"] = 0.24933916606795983
            self.w_os["res5c"] = 0.4320931831878144
            self.w_ot["conv1"] = 0.5169743140475447
            self.w_ot["res2c"] = 0.6733070050121481
            self.w_ot["res3d"] = 0.21573017541852707
            self.w_ot["res4f"] = 0.9378209392508551
            self.w_ot["res5c"] = 0.6055805286243681
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
            G_o.append(self.generate_style_image(activation[layer]))
            del activation[layer]

        self.models["resnet50"].conv1.register_forward_hook(self.get_activation('conv1')) # conv1
        self.models["resnet50"].layer1[2].conv3.register_forward_hook(self.get_activation('res2c')) # res2c
        self.models["resnet50"].layer2[3].conv3.register_forward_hook(self.get_activation('res3d')) # res3d
        self.models["resnet50"].layer3[5].conv3.register_forward_hook(self.get_activation('res4f')) # res4f
        self.models["resnet50"].layer4[2].conv3.register_forward_hook(self.get_activation('res5c')) # res5c
        output = self.models["resnet50"](source)

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
            output = self.models["resnet50"](target[i, :, :, :].unsqueeze(0))
            for idx, layer in enumerate(L):
                if len(G_t) < 5:
                    G_t.append(self.generate_style_image(activation[layer]) / target.shape[0])
                else:
                    G_t[idx] += self.generate_style_image(activation[layer]) / target.shape[0]
                del activation[layer]

        loss = torch.tensor(0, device=self.device)
        for i in range(len(M_o)):
            layer = L[i]
            loss = loss + self.w_os[layer] * torch.dist(M_o[i], M_s[i], 2) + self.alpha * self.w_ot[layer] * torch.dist(G_o[i], G_t[i], 2)

        # test_loss = Variable(loss, requires_grad=True)
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

            # model_ft = models.resnet50()
            model_ft =  ResNetMultiImageInput(models.resnet.Bottleneck, [3, 4, 6, 3])
            loaded = torch.load('model/pretrained/resnet50.pth')
            loaded['conv1.weight'] = torch.mean(loaded['conv1.weight'], dim=1).view(64, 1, 7, 7)
            model_ft.load_state_dict(loaded)
            # model_ft.load_state_dict(torch.load('model/pretrained/resnet50.pth'))
            self.set_parameter_requires_grad(model_ft, requires_grad)
            # num_ftrs = model_ft.fc.in_features
            # model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft
