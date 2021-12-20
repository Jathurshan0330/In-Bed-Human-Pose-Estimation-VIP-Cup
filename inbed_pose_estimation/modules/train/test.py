
import os
from detectron2.engine import DefaultPredictor

from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import random
import time

def test(dataset_name, cfg, n_samples_to_show=3, path_to_weights= "model_final.pth", full_weights_path=None):
  if full_weights_path==None:full_weights_path = os.path.join(cfg.OUTPUT_DIR, path_to_weights)
  else:pass
  cfg.MODEL.WEIGHTS =  full_weights_path # path to the model we just trained
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
  predictor = DefaultPredictor(cfg)

  dataset_dicts =  DatasetCatalog.get(dataset_name)


  for d in random.sample(dataset_dicts, n_samples_to_show):
      print(d["file_name"])
      im = cv2.imread(d["file_name"])
      outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
      v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)
      out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      pred_annot = out_pred.get_image()[:, :, ::-1]
      
      if 'rgb' in dataset_name:
        plt.figure(figsize= (8, 8))
        plt.imshow(cv2.cvtColor(pred_annot, cv2.COLOR_BGR2RGB), label = 'pred')
      else:
        plt.figure(figsize= (7, 3))
        plt.imshow(pred_annot, label = 'pred')

      savefilename = d["file_name"].replace('/', '@')

      if not os.path.isdir('test_results'):os.mkdir('test_results')
      
      plt.savefig(f'test_results/{str(time.time())}_{savefilename}')
      plt.show()
  return predictor
  