from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torchvision
import torch.nn as nn
import os


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
    'resnet18': [torchvision.models.resnet18, 256],
    'resnet34': [torchvision.models.resnet34, 256],
    'resnet50': [torchvision.models.resnet50, 1024],
    'resnet101': [torchvision.models.resnet101, 1024],
    'resnet152': [torchvision.models.resnet152, 1024],
}


class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.layers = num_layers
    self.dout_base_model =  model_urls['resnet' + str(self.layers)][1]
    model_name = '{}.pth'.format('resnet' + str(self.layers) + '_caffe')
    self.model_path = os.path.join(
    cfg.DATA_DIR, 'pretrained_model', model_name)

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = model_urls['resnet' + str(self.layers)][0](pretrained=True)
    if self.pretrained:
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        resnet.load_state_dict(
            {k: v for k, v in state_dict.items() if k in resnet.state_dict()})
        print('Done.')

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(resnet.fc.in_features, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(resnet.fc.in_features, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(resnet.fc.in_features, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
