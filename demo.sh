#!/bin/bash
model="./best_model_tr.pth"
video_read="tr_video_1.mp4"
video_write="tr_video_1_det.mp4"

python demo.py --load_name ${model} --video_read ${video_read} --video_write ${video_write} 
