
import scipy.io
import matplotlib.pyplot as plt
import glob
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog

from modules.utils.sorting_functions import *
from modules.data.data_utils import get_rainbow_colormap


def get_paired_data(train_dirs, type_='RGB', n_samples=None, bbox_style=None, img_folder_name= 'uncover'):
    img_id=1
    img_dict_list = []
    break_flag= False
    for train_dir in train_dirs:
        img_dirs = sorted(glob.glob(f'{train_dir}/{type_}/{img_folder_name}/*.png'), key = sort_by_img_name)
        joints_dir =  f'{train_dir}/joints_gt_{type_}.mat'
        joints_all = scipy.io.loadmat(joints_dir)['joints_gt']

        for img_dir in img_dirs: 
            
            img_number = img_dir[-10:-4]
            joints = joints_all[:,:,int(img_number)-1].T
            
            img= plt.imread(img_dir)
            if type_=='RGB':height, width, _ = img.shape
            else:height, width = img.shape

            #joints[:, 0]= joints[:,0]/width #x
            #joints[:, 1]= joints[:,1]/height #y

            #joints[:, 2]= np.ones((14,)).astype('float') #0, 1: represents occluded/ unoccluded of the point
            joints[:,0:2] = joints[:,0:2]-1 # according to inbed pose site

            joints_flatten= joints.flatten()
            
            x_max= joints[:, 0].max()
            x_min= joints[:, 0].min()

            y_max= joints[:, 1].max()
            y_min= joints[:, 1].min()

            bbox_ = [x_min, y_max, x_max, y_min]
            if bbox_style=='fixed_to_img_size':
              bbox_ = [0, height, width, 0]
            

            annotations= [{'bbox': bbox_, 
                           'bbox_mode':0,
                           'category_id':0,
                           'keypoints':joints_flatten}]

            img_dict = {'file_name':img_dir, 'height':height, 'width':width, 'image_id':img_id, 'annotations':annotations}
            img_dict_list.append(img_dict)
            
            if img_id%100==0:print(img_id)
            img_id+=1

            if n_samples!=None:
                if img_id>n_samples:
                    break_flag=True
                    break
        if break_flag==True:break
            
    return img_dict_list    

def set_metadata(dataset_name, colors_edgs):
    MetadataCatalog.get(dataset_name).keypoint_names = ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    MetadataCatalog.get(dataset_name).keypoint_flip_map = [('1', '6'),('2', '5'), ('3', '4'), ('7', '12'), ('8', '11'), ('9', '10')]

    MetadataCatalog.get(dataset_name).keypoint_connection_rules = [('1', '2', colors_edgs[0]), ('2', '3', colors_edgs[1]), ('3', '9', colors_edgs[2]), ('6', '5', colors_edgs[3]), ('5', '4', colors_edgs[4]), ('4', '10', colors_edgs[5]), ('7', '8', colors_edgs[6]), ('8', '9', colors_edgs[7]), ('12', '11', colors_edgs[8]), ('11', '10', colors_edgs[9]), ('9', '13', colors_edgs[10]), ('10', '13', colors_edgs[11]), ('13', '14', colors_edgs[12])]

def register_dataset(dataset_name, data_dirs, type_, n_samples, bbox_style, img_folder_name= 'uncover'):
    color_edgs= list(map(tuple, (255*get_rainbow_colormap()).astype('uint8')))
    def get_data(n_samples=n_samples):return get_paired_data(data_dirs, type_=type_, n_samples=n_samples, bbox_style=bbox_style, img_folder_name= img_folder_name)
    DatasetCatalog.register(dataset_name, get_data)
    set_metadata(dataset_name, color_edgs)