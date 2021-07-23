
import numpy as np
import cv2
import glob
import scipy.io
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

from modules.utils.sorting_functions import sort_by_img_name



def get_rainbow_colormap():
    rainbow_color_map= np.load('/content/gdrive/MyDrive/In-Bed-Human-Pose-Estimation(VIP-CUP)/Detectron2_supplementary_udith/mapvals_rainbow-gray_new.npy')[:-6]
    rainbow_color_map= rainbow_color_map*np.ones((20, rainbow_color_map.shape[0], 3))

    rainbow_color_map = cv2.resize(rainbow_color_map, (13, 20), interpolation=cv2.INTER_CUBIC)
    rainbow_color_map_array= rainbow_color_map[0]

    return rainbow_color_map_array

def show_keypoints(train_uncover_dirs, type_='RGB', train_uncover_dirs_id=0, img_dir_id=0):
    train_uncover_dir = train_uncover_dirs[train_uncover_dirs_id]

    img_dirs = sorted(glob.glob(f'{train_uncover_dir}/{type_}/uncover/*.png'), key = sort_by_img_name)
    joints_dir =  f'{train_uncover_dir}/joints_gt_{type_}.mat'
    joints_all = scipy.io.loadmat(joints_dir)['joints_gt']

    img_dir = img_dirs[img_dir_id]

    img_number = img_dir[-10:-4]

    joints = joints_all[:,:,int(img_number)-1].T
    
    img= plt.imread(img_dir)
    if type_=='RGB':height, width, _ = img.shape
    else:height, width = img.shape

    joints_flatten= joints.flatten()
    ###

    img_ = img.copy()


    for i in range(14):
        coord= joints[i]
        coord= (int(coord[0]), int(coord[1]))
        if type_=='RGB':img_ = cv2.circle(img_, coord, 10, (1,1,1), -1)
        if type_=='IR':img_ = cv2.circle(img_, coord, 2, (1), 2)


    
    plt.imshow(img_)
    plt.show()

def check_datasets(dataset_name= "paired_rgb_uncover"):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out= visualizer.draw_dataset_dict(d)
        #out = visualizer.draw_and_connect_keypoints(d['annotations'][0]['keypoints'].reshape(14, 3))
        cv2_imshow(out.get_image()[:, :, ::-1])