import torch
import numpy as np


class EvalMetrics:
    def __init__(self, pred, actual, n_classes=4):
        self.pred = torch.reshape(pred, actual.shape)
        self.actual = actual
        self.n_classes = n_classes

    def mask2onehot(self, mask):
        """
        Converts a segmentation mask (N,H,W) to (K,N,H,W) where K is the number of classes,
        N is the number of images, H is the number of rows and W is the number of columns
        """
        N, H, W = mask.shape
        _mask = torch.zeros([self.n_classes, N, H, W])
        for i in range(self.n_classes):
            _mask[i][mask == i] = 1
        return _mask

    def iou_coef(self, smooth=0):
        assert (self.pred.shape == self.actual.shape)

        axes = (2, 3)
        pred_onehot = self.mask2onehot(self.pred)
        actual_onehot = self.mask2onehot(self.actual)

        intersection = torch.sum(torch.abs(pred_onehot * actual_onehot), dim=axes)
        union = torch.sum(pred_onehot, dim=axes) + torch.sum(actual_onehot, dim=axes) - intersection
        iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
        return torch.mean(iou)

    def dice_coef(self, smooth=0):
        assert (self.pred.shape == self.actual.shape)

        axes = (2, 3)
        pred_onehot = self.mask2onehot(self.pred)
        actual_onehot = self.mask2onehot(self.actual)

        intersection = torch.sum(pred_onehot * actual_onehot, dim=axes)
        union = torch.sum(pred_onehot, dim=axes) + torch.sum(actual_onehot, dim=axes)
        dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
        return torch.mean(dice)
