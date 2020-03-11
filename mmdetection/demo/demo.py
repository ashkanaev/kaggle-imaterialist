from mmdet.apis import init_detector, inference_detector, show_result, result2dict
import json
import mmcv
import glob
from random import randint
import cv2
import os
import csv

with open('/home/aiex/repos/ifashion2/data/label_descriptions.json') as f:
  data = json.load(f)

categs = ['-'.join(i['name'].split(',')) for i in data['categories']]

config_file = '/home/aiex/repos/ifashion/configs/inference_conf.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/aiex/Downloads/sota_imat_epoch_15.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

test_images = sorted(glob.glob("/media/aiex/Новый том/T/_Datasets/Modis/test_images/MODIS_DRESS_ATTR/*jpg"))
# cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('demo', 600, 200)
retval = {}
for i_img in test_images:

    output_name = os.path.join('/home/aiex/repos/ifashion2/data/vis', os.path.basename(i_img))
    result = inference_detector(model, i_img)
    # show_result(i_img, result, categs, out_file=output_name)

    res_dict = result2dict(i_img, result, categs, out_file=output_name)
    if res_dict:
        retval[os.path.basename(i_img)] = {'poly': res_dict[1], 'labels': res_dict[2]}

w = csv.writer(open("modis.csv", "w"))
for key, val in retval.items():
    w.writerow([key, val])
with open('output_modis.json', 'w') as fp:
    json.dump(retval, fp)
    # show_result(i_img, result, categs, out_file=output_name)

# print(result)
# show the results
# show_result_pyplot(img, result, model.CLASSES)
