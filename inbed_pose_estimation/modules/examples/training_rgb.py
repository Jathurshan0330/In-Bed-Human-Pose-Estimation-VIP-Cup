
import sys
sys.path.append('/content/inbed_pose_estimation')

from modules.utils.sorting_functions import sort_by_subject
import glob
import os

from modules.data.data_utils import get_rainbow_colormap
from modules.data.create_detectron_datasets import register_dataset
import numpy as np

from modules.train.train import train
from modules.train.test import test

import argparse

type_ = ['IR', 'RGB'][1]


########################################################################################################
parser = argparse.ArgumentParser(description='train with TL')
parser.add_argument('--TL_model_dir', type=str, default=None, required=False)
parser.add_argument('--img_folder_name', type=str, default='uncover', required=False)

args = parser.parse_args()
TL_model_dir= args.TL_model_dir
img_folder_name= args.img_folder_name

print(f'Transfer Learned from : {TL_model_dir}')
#########################################################################################################

train_uncover=  []
for data_dir in glob.glob('In-Bed-Human-Pose-Estimation(VIP-CUP)/train/*'):
    if int(data_dir[-5:])<=30:train_uncover.append(data_dir)
train_uncover_dirs= sorted(train_uncover, key=sort_by_subject)
#########################################################################################################

idx=np.random.randint(low= 5000, high=1000000000000)

if img_folder_name=='uncover':
  train_dataset_name = f"paired_{type_}_uncoverv{idx}_train"
  test_dataset_name = f"paired_{type_}_uncoverv{idx}_test"

else:
  train_dataset_name = f"paired_{type_}_generated_uncoverv{idx}_train"
  test_dataset_name = f"paired_{type_}_generated_uncoverv{idx}_test"


n_samples= None
bbox_style= 'fixed_to_img_size' #None can be used
register_dataset(train_dataset_name, train_uncover_dirs[:24], type_, n_samples, bbox_style = bbox_style, img_folder_name= img_folder_name)
register_dataset(test_dataset_name, train_uncover_dirs[24:], type_, n_samples, bbox_style = bbox_style, img_folder_name= img_folder_name)

##########################################################################################################

cfg = train(train_dataset_name, test_dataset_name, max_iter = 5000, TL_model_dir= TL_model_dir)
predictor = test(test_dataset_name, cfg, n_samples_to_show=3, path_to_weights = f'model_final.pth')

print(f'SAVED_MODEL_DIR : {os.path.join(cfg.OUTPUT_DIR, "model_final.pth")}')