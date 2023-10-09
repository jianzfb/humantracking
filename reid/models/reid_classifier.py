import torch
import torch.nn.functional as F
from antgo.framework.helper.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck
from antgo.framework.helper.models.classification.model.base import BaseClassifier
import torchvision
from .model import *
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning
from .circle_loss import CircleLoss, convert_label_to_similarity
from .instance_loss import InstanceLoss


@CLASSIFIERS.register_module()
class ReidClassifier(BaseClassifier):
    def __init__(self,
                 train_cfg=None,
                 init_cfg=None, **kwargs):
        super(BaseClassifier, self).__init__(init_cfg)
        model_ft = torchvision.models.resnet50(pretrained=True)
        class_num = train_cfg.get('class_num')
        droprate = train_cfg.get('droprate', 0.5)
        stride = train_cfg.get('stride', 2)
        circle = train_cfg.get('circle', False)
        linear_num = train_cfg.get('linear_num', 512)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.model.fc = nn.Sequential()
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)
        self.criterion = nn.CrossEntropyLoss()

        self.criterion_arcface = None
        if train_cfg.get('arcface', False):
            self.criterion_arcface = losses.ArcFaceLoss(num_classes=class_num, embedding_size=512)
        self.criterion_cosface = None
        if train_cfg.get('cosface', False):
            self.criterion_cosface = losses.CosFaceLoss(num_classes=class_num, embedding_size=512)
        self.criterion_circle = None
        if train_cfg.get('circle', False):
            self.criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
        self.criterion_triplet = None
        if train_cfg.get('triplet', False):
            self.miner = miners.MultiSimilarityMiner()
            self.criterion_triplet = losses.TripletMarginLoss(margin=0.3)
        self.criterion_lifted = None
        if train_cfg.get('lifted', False):
            self.criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
        self.criterion_contrast = None
        if train_cfg.get('contrast', False):
            self.criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.criterion_instance = None
        if train_cfg.get('instance', False):
            self.criterion_instance = InstanceLoss(gamma = opt.ins_gamma)
        self.criterion_sphere = None
        if train_cfg.get('sphere', False):
            self.criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=512, margin=4)

    def forward_train(self, image, label=None, **kwargs):
        """Forward computation during training.
        Args:
            image (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                should be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        logits, feature = self.classifier(x)
        if label is None:
            return feature

        loss = self.criterion(logits, label)
        losses = {
            'loss': loss
        }

        sm = nn.Softmax(dim=1)
        log_sm = nn.LogSoftmax(dim=1)
        return_feature = self.criterion_arcface or \
            self.criterion_cosface or \
            self.criterion_circle or \
            self.criterion_triplet or \
            self.criterion_lifted or \
            self.criterion_contrast or \
            self.criterion_instance or self.criterion_sphere

        if return_feature:
            fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
            feature = feature.div(fnorm.expand_as(feature))
            _, preds = torch.max(logits.data, 1)
            now_batch_size = image.shape[0]
            aux_loss = 0
            if self.criterion_arcface is not None:
                aux_loss +=  self.criterion_arcface(feature, label)/now_batch_size
            if self.criterion_cosface is not None:
                aux_loss +=  self.criterion_cosface(feature, label)/now_batch_size
            if self.criterion_circle is not None:
                aux_loss +=  self.criterion_circle(*convert_label_to_similarity(feature, label))/now_batch_size
            if self.criterion_triplet is not None:
                hard_pairs = self.miner(feature, label)
                aux_loss +=  self.criterion_triplet(feature, label, hard_pairs) #/now_batch_size
            if self.criterion_lifted is not None:
                aux_loss +=  self.criterion_lifted(feature, label) #/now_batch_size
            if self.criterion_contrast is not None:
                aux_loss +=  self.criterion_contrast(feature, label) #/now_batch_size
            if self.criterion_instance is not None:
                aux_loss += self.criterion_instance(feature) /now_batch_size
            if self.criterion_sphere is not None:
                aux_loss +=  self.criterion_sphere(feature, label)/now_batch_size

            losses.update({
                'aux_loss': aux_loss
            })
        return losses

    def simple_test(self, image, image_meta=None, **kwargs):
        """Test without augmentation."""
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        ff = x[1]
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return {
            'feature': ff
        }

    def onnx_export(self, image):
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)

        ff = x[1]
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff