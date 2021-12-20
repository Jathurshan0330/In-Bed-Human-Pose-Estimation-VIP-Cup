
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from modules.evaluation.visualizer_new import Visualizer as Visualizer_new


def predict_poses(images, cfg, dataset_name, full_model_dir, score_thres= 0.7, vis = 'old'):
    cfg.MODEL.WEIGHTS = full_model_dir # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    pred_images = []
    pred_annots = []
    for im in images:
        outputs = predictor(im)
        if vis=='new':v = Visualizer_new(im[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)
        elif vis == 'old':v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)


        preds= outputs["instances"].to("cpu")
        preds.scores = torch.ones_like(preds.scores)

        out_pred = v.draw_instance_predictions(preds)
        pred_annot = out_pred.get_image()[:, :, ::-1]

        pred_images.append(pred_annot)
        pred_annots.append(outputs["instances"].to("cpu").pred_keypoints)
    return pred_images, pred_annots

def get_data_point(data, idx): # idx < n_samples
    height, width = data[0]['height'], data[0]['width']

    file_name = data[idx]['file_name']
    keypoints= data[idx]['annotations'][0]['keypoints'].reshape(1, 14, 3)
    return file_name, keypoints, height, width # keypoints.shape: (1, 14, 3) where value of 3rd dim=1

def get_data_points(data, indices):
    keypoint_list= []
    file_name_list =[]
    height_list =[]
    width_list= []
    for idx in indices:
        file_name, keypoints, height, width = get_data_point(data, idx)

        keypoint_list.append(keypoints)
        file_name_list.append(file_name)
        height_list.append(height)
        width_list.append(width)

    height, width = height_list[0], width_list[0]
    assert (np.array(height_list)==height).all(), 'Height is not constant, Could not caoculate accuracy'
    assert (np.array(width_list)==width).all(), 'Width is not constant, Could not caoculate accuracy'

    return file_name_list, np.array(keypoint_list)[:, 0], height, width



def get_annotated_ground_truth(img_dir, joints):    # joints: (14,3) 
    img= plt.imread(img_dir)

    if len(img.shape)==3:type_='RGB'
    else:type_='IR'

    
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

    return img_