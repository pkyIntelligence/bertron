import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from detectron2.config import get_cfg
from detectron2.engine import DefaultMultiImgPredictor
from detectron2.utils.logger import setup_logger

from transformers import *

from Capstone.data_utils import *


MAX_TGT_LENGTH = 67
CPU_DEVICE = torch.device('cpu')
GPU_DEVICE = torch.device('cuda:0')
TRAIN_IMGS_DIR = os.path.join('Capstone', 'coco', 'train2014')
CAPTIONS_FILE = os.path.join('Capstone', 'coco', 'annotations', 'captions_train2014.json')
MASK_PROB = 0.15

detectron_cfg = get_cfg()
detectron_cfg.merge_from_file('Capstone/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x_FC7.yaml')
detectron_cfg.merge_from_list(['MODEL.WEIGHTS', 'Capstone/model_weights/model_final_f6e8b1.pkl'])
detectron_cfg.freeze()
logger = setup_logger(os.path.join('logs', 'log.txt'))
max_detections = detectron_cfg['TEST']['DETECTIONS_PER_IMAGE']
detectron_predictor = DefaultMultiImgPredictor(detectron_cfg)

max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
state_dict = torch.load(os.path.join('Capstone', 'model_weights', 'model.30.bin'))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

train_ds = CocoCaptions(TRAIN_IMGS_DIR, CAPTIONS_FILE)
train_dl = DataLoader(train_ds, batch_size=1, num_workers=4, collate_fn=detectron_collate, pin_memory=True)


for i, data in enumerate(tqdm(train_dl), 0):
    # get the inputs; data is a list of [imgs, captions]
    imgs, captions = data

    with torch.no_grad():  # Data preprocessing/pipelining
        imgs = [np.array(img)[:, :, ::-1] for img in imgs]
        preds = detectron_predictor(imgs)

        vis_feats, vis_pe = get_img_tensors(preds[0], max_detections)
        # Move back to CPU for disk writing
        vis_feats = vis_feats.to(CPU_DEVICE)
        vis_pe = vis_pe.to(CPU_DEVICE)
        # write to visual features and positional encoding file
        with open(f'D:\\datasets\\coco2014_features_cap\\vis_feat_pe_{i:06}.pkl', 'wb') as vis_f:
            pkl.dump(vis_feats, vis_f, protocol=pkl.HIGHEST_PROTOCOL)
            pkl.dump(vis_pe, vis_f, protocol=pkl.HIGHEST_PROTOCOL)

        os.mkdir(f'D:\\datasets\\coco2014_features_cap\\vis_feat_pe_{i:06}')

        captions = captions[0]
        for j, caption in enumerate(captions, 0):
            input_ids, segment_ids, attn_mask, masked_ids, masked_pos, masked_weights = \
                prepare_bert_caption_train(tokenizer, vis_feats.shape[0], caption, max_input_len,
                                           round(MAX_TGT_LENGTH*MASK_PROB), MASK_PROB)

            # write to captions file
            with open(f'D:\\datasets\\coco2014_features_cap\\vis_feat_pe_{i:06}\\caption_{j:02}.pkl', 'wb') as cap_f:
                pkl.dump(input_ids, cap_f, protocol=pkl.HIGHEST_PROTOCOL)
                pkl.dump(segment_ids, cap_f, protocol=pkl.HIGHEST_PROTOCOL)
                pkl.dump(attn_mask, cap_f, protocol=pkl.HIGHEST_PROTOCOL)
                pkl.dump(masked_ids, cap_f, protocol=pkl.HIGHEST_PROTOCOL)
                pkl.dump(masked_pos, cap_f, protocol=pkl.HIGHEST_PROTOCOL)
                pkl.dump(masked_weights, cap_f, protocol=pkl.HIGHEST_PROTOCOL)
