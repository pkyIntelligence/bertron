import torch

import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultMultiImgPredictor
from detectron2.data.detection_utils import read_image

from transformers.configuration_bert import *
from transformers.modeling_bert import *
from transformers.tokenization_bert import *

from Capstone.data_utils import *


DATA_DIR = '/home/pky/Datasets'
MODEL_DIR = 'model_weights'
MAX_TGT_LENGTH = 67
CPU_DEVICE = torch.device('cpu')
GPU_DEVICE = torch.device('cuda:0')
TRAIN_IMGS_DIR = os.path.join(DATA_DIR, 'coco', 'train2014')
CAPTIONS_FILE = os.path.join(DATA_DIR, 'coco', 'annotations', 'captions_train2014.json')
MASK_PROB = 0.15


detectron_cfg = get_cfg()
detectron_cfg.merge_from_file('Capstone/detectron2/configs/COCO-Detection/faster_rcnn_X_101_64x4d_FPN_2x_vlp.yaml')
detectron_cfg.merge_from_list(['MODEL.WEIGHTS', 'Capstone/model_weights/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl'])
detectron_cfg.freeze()
max_detections = detectron_cfg['TEST']['DETECTIONS_PER_IMAGE']
detectron_predictor = DefaultMultiImgPredictor(detectron_cfg)
img1 = read_image(os.path.join(TRAIN_IMGS_DIR, 'COCO_train2014_000000003000.jpg'), format="BGR")
img2 = read_image(os.path.join(TRAIN_IMGS_DIR, 'COCO_train2014_000000013000.jpg'), format="BGR")
preds = detectron_predictor([img1, img2])


with torch.no_grad():

    max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
    state_dict = torch.load(os.path.join('Capstone', 'model_weights', 'model.30.bin'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

    fc_dim = detectron_cfg['MODEL']['ROI_BOX_HEAD']['FC_DIM']
    num_classes = detectron_cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']

    vis_feats, vis_pe = get_img_tensors(preds[1], fc_layer=1, fc_dim=fc_dim, num_classes=num_classes,
                                        max_detections=max_detections)
    import ipdb; ipdb.set_trace()
    input_ids, segment_ids, position_ids, attn_mask = \
        prepare_bert_caption_inf(tokenizer.convert_tokens_to_ids, vis_feats.shape[0], max_detections, max_input_len)

    # max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

    bert_config = BertForImg2TextConfig(vis_feat_dim=fc_dim, vis_classes=num_classes+1, type_vocab_size=6)

    model = BertForImg2Txt.from_pretrained(
                pretrained_model_name_or_path='bert-base-cased',
                config=bert_config,
                state_dict=state_dict,
                num_labels=2,
                len_vis_input=100,
            ).to(GPU_DEVICE)
    model.eval()

    vis_feats = vis_feats.unsqueeze(0).to(GPU_DEVICE)
    vis_pe = vis_pe.unsqueeze(0).to(GPU_DEVICE)
    input_ids = input_ids.unsqueeze(0).to(GPU_DEVICE)
    segment_ids = segment_ids.unsqueeze(0).to(GPU_DEVICE)
    position_ids = position_ids.unsqueeze(0).to(GPU_DEVICE)
    attn_mask = attn_mask.unsqueeze(0).to(GPU_DEVICE)

    traces = model(vis_feats=vis_feats, vis_pe=vis_pe, input_ids=input_ids, token_type_ids=segment_ids,
                   position_ids=position_ids, attention_mask=attn_mask, task_idx=3)

    tokenizer.convert_ids_to_tokens(traces[0][0])
