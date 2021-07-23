
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import os


def train(dataset_name_train, dataset_name_test, max_iter= 100, TL_model_dir= None):
    cfg= get_cfg_train(dataset_name_train, dataset_name_test, max_iter, TL_model_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #cfg.MODEL.KEYPOINT_ON= True

    trainer = DefaultTrainer(cfg) 
    
    for name, param in trainer.model.named_parameters():
        if name.split('.')[0]=='backbone':param.requires_grad=False

    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg

def get_cfg_train(dataset_name_train, dataset_name_test, max_iter= 100, TL_model_dir= None):
    cfg = get_cfg()  
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name_train,)
    cfg.DATASETS.TEST = (dataset_name_test,)

    cfg.DATALOADER.NUM_WORKERS = 2
    if TL_model_dir==None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = os.path.join(TL_model_dir)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter #300    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # human

    ###
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    ###
    cfg.OUTPUT_DIR= f'./output/{dataset_name_train}'
    return cfg

def get_cfg_test(dataset_name_test):
    cfg= get_cfg_train(dataset_name_train= None, dataset_name_test = dataset_name_test, TL_model_dir= None)
    return cfg