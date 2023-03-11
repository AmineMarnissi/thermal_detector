# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import cv2
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def vis_detections(im, class_name, dets, thresh):
    """Visual debugging of detections."""
    compte_person = 0
    compte_car = 0
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        print(score)
        image = im.copy()
        # Display parameter
        font_scale = 0.8
        font = cv2.FONT_HERSHEY_PLAIN
        if class_name!="dog":
            if score > thresh:
                if class_name=="person":
                    color = (115, 144, 32)
                    compte_person = compte_person + 1
                elif class_name=="car":
                    color = (196,181,74)
                    compte_car = compte_car + 1
                elif class_name=="bicycle":
                    color = (102,217,255)
                else:
                    color= (102,217,255)
                alpha = 0.6
                cv2.rectangle(im, bbox[0:2], bbox[2:4], color, -1)
                cv2.addWeighted(image, alpha, im, 1 - alpha,0, im)
                cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
                print(score)
                #set some text
                text = class_name+':'+str(round(score,2))
                #get the width and height of the text box
                (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
                cv2.rectangle(im, bbox[0:2], (bbox[0]+text_width+2,bbox[1]-text_height-2), color, -1)
                cv2.putText(im,text, (bbox[0], bbox[1]), font, font_scale, (255,255, 255), thickness=2)

    cv2.putText(
        im, #numpy array on which text is written
        "Car: "+str(compte_car), #text
        (10,50), #position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX, #font family
        0.6, #font size
        (255, 255, 255), #font color
        1) #font stroke
    
    cv2.putText(
        im, #numpy array on which text is written
        "Person: "+str(compte_person), #text
        (10,70), #position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX, #font family
        0.6, #font size
        (255, 255, 255), #font color
    1) #font stroke    
    return im

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box