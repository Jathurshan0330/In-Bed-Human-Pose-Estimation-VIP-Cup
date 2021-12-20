


import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.evaluation.metrics import accuracy
from modules.evaluation.eval_utils import get_data_points, predict_poses

from modules.evaluation.eval_utils import get_annotated_ground_truth


def get_accuracy(data, indices, cfg, test_dataset_name,  full_model_dir1,  full_model_dir2, label_model1, label_model2, show_results_indices= None, vis = 'old'):
    img_names, ground_annots, height, width = get_data_points(data, indices) # ground annot: (m, 14, 3)
  
    images_ir= []
    for img_name in img_names:
        images_ir.append(cv2.imread(img_name))

    pred_images_model1, pred_annots_model1 = predict_poses(images_ir, cfg, dataset_name=  test_dataset_name, full_model_dir= full_model_dir1, vis =vis)
    pred_images_model2, pred_annots_model2 = predict_poses(images_ir, cfg, dataset_name=  test_dataset_name, full_model_dir= full_model_dir2, vis =vis)
    # pred_annots__model1, TL.shape: (m, 1, 14, 3)

    pred_annots_model1 = np.array([val.numpy() for val in pred_annots_model1])[:, 0] # (m, 14, 3)
    pred_annots_model2 = np.array([val.numpy() for val in pred_annots_model2])[:, 0] # (m, 14, 3)
    
    acc_model1 = accuracy(output = pred_annots_model1[:, :, :2], target = ground_annots[:, :, :2], thr=0.5, h= height, w= width)
    acc_model2 = accuracy(output = pred_annots_model2[:, :, :2], target = ground_annots[:, :, :2], thr=0.5, h= height, w= width)


    print(f'Acc({label_model1}) : {acc_model1[1]}')
    print(f'Acc({label_model2}) : {acc_model2[1]}')
    if show_results_indices!=None:
        for show_results_idx in show_results_indices:
            acc_model1_single = accuracy(output = pred_annots_model1[show_results_idx:show_results_idx+1, :, :2], target = ground_annots[show_results_idx:show_results_idx+1, :, :2], thr=0.5, h= height, w= width)[1]
            acc_model2_single = accuracy(output = pred_annots_model2[show_results_idx:show_results_idx+1, :, :2], target = ground_annots[show_results_idx:show_results_idx+1, :, :2], thr=0.5, h= height, w= width)[1]


            plt.figure(figsize= (8,7))
            plt.subplot(1,3,1)
            plt.imshow(pred_images_model1[show_results_idx])
            plt.title(f'{label_model1}:: acc : {np.round(acc_model1_single, 3)}')

            plt.subplot(1,3,2)
            plt.imshow(pred_images_model2[show_results_idx])
            plt.title(f'{label_model2}:: acc : {np.round(acc_model2_single, 3)}')

            plt.subplot(1, 3, 3)
            plt.imshow(get_annotated_ground_truth(img_names[show_results_idx], ground_annots[show_results_idx]))
            plt.show()