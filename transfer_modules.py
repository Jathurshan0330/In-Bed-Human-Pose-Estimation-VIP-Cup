
import shutil
import os

def send_modules():
  if not os.path.isdir('/content/inbed_pose_estimation/modules'):
    print('no modules to send !!!')
    return 0

  try:shutil.rmtree('/content/gdrive/My Drive/CODES_inbedpose_estimation/inbed_pose_estimation/modules')
  except:pass
  shutil.copytree('/content/inbed_pose_estimation/modules', '/content/gdrive/My Drive/CODES_inbedpose_estimation/inbed_pose_estimation/modules')
  print(f'modules send to : /content/gdrive/My Drive/CODES_inbedpose_estimation/inbed_pose_estimation/modules')
def recieve_modules():
  try:shutil.rmtree('/content/inbed_pose_estimation/modules')
  except:pass
  shutil.copytree('/content/gdrive/My Drive/CODES_inbedpose_estimation/inbed_pose_estimation/modules', '/content/inbed_pose_estimation/modules')
  print('modules recieved to : /content/inbed_pose_estimation/modules')