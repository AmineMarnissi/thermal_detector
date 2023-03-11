import argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="",
                        type=str)
    parser.add_argument('--video_read', dest='video_read',
                    help='path to video read', default="",
                    type=str)
    parser.add_argument('--video_write', dest='video_write',
                    help='path to load models', default="",
                    type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default= True)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=0, type=int)
    args = parser.parse_args()
    return args

def set_dataset_args(args, test=False):
    if test:
        args.imdb_name = "flir_tr_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES','20']
    return args
