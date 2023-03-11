# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email.policy import strict

#import _init_paths
import os
import sys
import numpy as np
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
from torchvision.ops import nms
from config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes,bbox_transform_inv
from model.utils.net_utils import vis_detections, _get_image_blob
from model.utils.parser_func import parse_args, set_dataset_args
from model.faster_rcnn.resnet import resnet

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  args = set_dataset_args(args, test=True)

  cfg_from_file("model_tr.yml")
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  _classes = np.asarray(['__background__',  # always index 0
                          'person','car','_','_'])
  fasterRCNN = resnet(_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  fasterRCNN.create_architecture()
  
  print("load checkpoint %s" % (args.load_name))
  if args.cuda > 0:
    checkpoint = torch.load(args.load_name)
  else:
    checkpoint = torch.load(args.load_name, map_location=(lambda storage, loc: storage))
  
  fasterRCNN.load_state_dict(checkpoint['model'],strict=False)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

  if args.cuda > 0:
    cfg.CUDA = True
    fasterRCNN.cuda()
  fasterRCNN.eval()
  start = time.time()
  thresh = 0.3
  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(args.video_read)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(args.video_write,cv2.VideoWriter_fourcc('M','J','P','G'), 8, (frame_width,frame_height))
    num_images = 0
  while (num_images >= 0):
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1
      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      im = im_in
      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.resize_(1, 1, 5).zero_()
      num_boxes.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data
      # Optionally normalize targets by a precomputed mean and stdev
      if args.cuda > 0:
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
      else:
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
      box_deltas = box_deltas.view(1, -1, 4 * len(_classes))
      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      pred_boxes /= im_scales[0]
      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      im2show = np.copy(im)
      for j in range(1, len(_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = keep = nms(cls_boxes[order, :],
                           cls_scores[order], 0.55)
            cls_dets = cls_dets[keep.view(-1).long()]
            im2show = vis_detections(im2show, _classes[j], cls_dets.cpu().numpy(), thresh=0.8)
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic
      out.write(im2show)
      cv2.imshow("frame", im2show)
      total_toc = time.time()
      total_time = total_toc - total_tic
      frame_rate = 1 / total_time
      print('Frame rate:', frame_rate)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()
