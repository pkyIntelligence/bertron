from detectron2.config import get_cfg
from detectron2.engine import DefaultMultiImgPredictor
from detectron2.data.detection_utils import read_image

from transformers import *

from Capstone.data_utils import *


MAX_TGT_LENGTH = 67
CPU_DEVICE = torch.device('cpu')
GPU_DEVICE = torch.device('cuda:0')
TRAIN_IMGS_DIR = os.path.join('Capstone', 'coco', 'train2014')
CAPTIONS_FILE = os.path.join('Capstone', 'coco', 'annotations', 'captions_train2014.json')
MASK_PROB = 0.15
DATA_DIR = r'D:\datasets\coco2014_features_cap'
MODEL_DIR = r'D:\models'


detectron_cfg = get_cfg()
detectron_cfg.merge_from_file('Capstone/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x_FC7.yaml')
detectron_cfg.merge_from_list(['MODEL.WEIGHTS', 'Capstone/model_weights/model_final_f6e8b1.pkl'])
detectron_cfg.freeze()
max_detections = detectron_cfg['TEST']['DETECTIONS_PER_IMAGE']
detectron_predictor = DefaultMultiImgPredictor(detectron_cfg)
img1 = read_image(os.path.join('Capstone', 'test_images', '12283150_12d37e6389_z.jpg'), format="BGR")
img2 = read_image(os.path.join('Capstone', 'test_images', '25691390_f9944f61b5_z.jpg'), format="BGR")
preds = detectron_predictor([img1, img2])


max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
state_dict = torch.load(os.path.join('Capstone', 'model_weights', 'model.30.bin'))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

vis_feats, vis_pe = get_img_tensors(preds[0], max_detections)

input_ids, segment_ids, position_ids, attn_mask = \
    prepare_bert_caption_inf(tokenizer.convert_tokens_to_ids, vis_feats.shape[0], max_detections, max_input_len)

# max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
state_dict = torch.load(os.path.join('Capstone', 'model_weights', 'model.30.bin'))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

model = BertForImg2Txt.from_pretrained(
            pretrained_model_name_or_path='bert-base-cased',
            config=None,
            state_dict=state_dict,
            num_labels=2,
            len_vis_input=100,
            type_vocab_size=6,
        ).to(GPU_DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'coco_caption_bert2.pt')))
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
