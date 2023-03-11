#!/bin/bash
model="./best_model_tr.pth"

python demo.py --load_name ${model} --video_read $1 --video_write $2
