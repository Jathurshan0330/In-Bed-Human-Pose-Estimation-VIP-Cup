
import subprocess, os, shutil, glob
import argparse
import cv2

def refine_dir(resultdir, model_name, resize_shape= None):
  important = glob.glob(f'{resultdir}/{model_name}/test_latest/images/*_fake.png')

  for img_dir in glob.glob(f'{resultdir}/{model_name}/test_latest/images/*'):
    if img_dir not in important:
      os.remove(img_dir)
    else:
      img_id = img_dir.split('/')[-1].split('_')[1]
      new_img_dir = '/'.join(img_dir.split('/')[:-4]) + f'/image_{img_id}.png'

      if resize_shape== None:
        os.rename(img_dir, new_img_dir )
      else:
        cv2.imwrite(new_img_dir, cv2.resize(cv2.imread(img_dir), resize_shape))

  shutil.rmtree(f'{resultdir}/{model_name}')

def remove_cyclegan_results(remove_folder= 'cover1_cyclegan'): #generate covered data through cyclegan and save it in [DATA_PATH]/result_folder
  for id_ in sorted(os.listdir(f'/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/'), key= int)[:30]: #only train set

    remove_path = f'/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/{id_}/IR/{remove_folder}'
    try:
      shutil.rmtree(remove_path)
      print(f'removed folder : {remove_path}')
    except:
      pass
    

def get_cyclegan_results(model_name= 'InbedPose_CyleGAN', result_folder= 'cover1_cyclegan', remove_folder = None, resize_shape= None): #generate covered data through cyclegan and save it in [DATA_PATH]/result_folder
  if remove_folder!=None:
    remove_cyclegan_results(remove_folder= remove_folder)


  if result_folder!=None:
    for id_ in sorted(os.listdir(f'/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/'), key= int)[:30]: #only train set

      try:
        shutil.rmtree(f'/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/{id_}/IR/{result_folder}')
      except:
        pass
      
      command = f"python test.py --dataroot '/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/{id_}/IR/uncover' --name {model_name} --model test --results_dir '/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/{id_}/IR/{result_folder}' --no_dropout"
      print(f'cmd : {command}')
      subprocess.check_output(command, shell=True)
      refine_dir(f'/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/{id_}/IR/{result_folder}', model_name= model_name, resize_shape= resize_shape)


parser = argparse.ArgumentParser(description='cyclegan')
parser.add_argument('--model_name', type=str, required=False, default=None)
parser.add_argument('--resize_shape', type=str, required=False, default=None) ## eg: (100, 20)
parser.add_argument('--result_folder', type=str, required=False, default=None)
parser.add_argument('--remove_folder', type=str, required=False, default=None)


args = parser.parse_args()
resize_shape = eval(str(args.resize_shape))

get_cyclegan_results(model_name= args.model_name, result_folder= args.result_folder, remove_folder= args.remove_folder, resize_shape= resize_shape)