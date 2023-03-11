#!/bin/bash
model="./best_model_tr.pth"
video_read=$0
video_write=$1

python demo.py --load_name ${model} --video_read ${video_read} --video_write ${video_write} 
