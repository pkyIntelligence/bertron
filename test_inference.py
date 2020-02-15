import os
import torch

from captioner import Captioner


DATA_DIR = '/home/pky/Datasets'
MODEL_DIR = 'model_weights'
CONFIG_DIR = '/home/pky/Configs'
OBJ_VOCAB_FILE = 'vocab/objects_vocab.txt'

CPU_DEVICE = torch.device('cpu')
GPU_DEVICE = torch.device('cuda:0')
TRAIN_IMGS_DIR = os.path.join(DATA_DIR, 'coco', 'train2014')

detector_cfg_path = os.path.join(CONFIG_DIR, 'detectron2/COCO-Detection/faster_rcnn_X_101_64x4d_FPN_2x_vlp.yaml')
detector_weights_path = os.path.join(MODEL_DIR, 'detectron/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl')
bert_cfg_path = 'VLP/configs/bert_for_captioning.json'
bert_weights_path = os.path.join(MODEL_DIR, 'bert/model.19.bin')

captioner = Captioner(detector_cfg_path, detector_weights_path, bert_cfg_path, bert_weights_path, OBJ_VOCAB_FILE,
                      CPU_DEVICE, GPU_DEVICE)

"https://farm2.staticflickr.com/1080/1301049949_532835a8b5_z.jpg"
img_path = os.path.join(TRAIN_IMGS_DIR, 'COCO_train2014_000000018000.jpg')
captioner("https://farm3.staticflickr.com/2468/4016197796_592dfd3183_z.jpg")
