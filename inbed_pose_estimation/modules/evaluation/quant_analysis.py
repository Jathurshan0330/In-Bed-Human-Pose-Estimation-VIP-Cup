
import numpy as np
from detectron2.data import DatasetCatalog
from modules.train.train import get_cfg_test
from modules.data.create_detectron_datasets import register_dataset
from modules.evaluation.accuracy_detectron import get_accuracy


def quantitative_analysis(data_dirs, type_, img_folder_name, n_samples= None, bbox_style= 'fixed_to_img_size', full_model_dir1=None, full_model_dir2=None, label_model1=None, label_model2=None, vis_type= 'old', show_n_results= 10):
    '''
    data_dirs:: eg: ['In-Bed-Human-Pose-Estimation(VIP-CUP)/train/00025', ...]
    type_:: eg: 'IR'
    img_folder_name:: eg: 'uncover'

    -> img_dirs obtained as: f"{data_dirs[i]}/{type_}/{img_folder_name}"
    '''
    
    test_dataset_name = f"test_{np.random.randint(low= 5000, high=1000000000000)}"
    register_dataset(test_dataset_name, data_dirs, type_, n_samples, bbox_style = bbox_style, img_folder_name= img_folder_name)
    #########################

    data = DatasetCatalog.get(test_dataset_name)
    cfg =  get_cfg_test(test_dataset_name)
    get_accuracy(data, range(len(data)), cfg, test_dataset_name,  full_model_dir1,  full_model_dir2, label_model1, label_model2, show_results_indices= range(show_n_results), vis = vis_type)