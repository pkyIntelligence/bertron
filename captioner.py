from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image

from VLP.pytorch_pretrained_bert.modeling import BertConfig, BertForSeq2SeqDecoder
from VLP.pytorch_pretrained_bert.tokenization import BertTokenizer

from data_utils import *


class Captioner:
    """
    A Captioner, some sort of detector or image feature extractor combined with a decoder to generate the caption
    """
    def __init__(self, detector_cfg_path, detector_weights_path, bert_cfg_path, bert_weights_path, device, fc_layer=0,
                 max_caption_length=67):
        """
        args:
            detector_cfg_path: path to the detector config
            detector_weights_path: path to the detector weights
            bert_cfg_path: path to the bert decoder config
            bert_weights_path: path to the bert decoder weights
            device: The device to run the computations, currently only supports GPU devices
            fc_layer: the fully connected layer from the detector to extract features from, 0-indexed
        """
        self.device = device

        detector_cfg = get_cfg()
        detector_cfg.merge_from_file(detector_cfg_path)
        detector_cfg.merge_from_list(['MODEL.WEIGHTS', detector_weights_path])
        detector_cfg.freeze()

        self.detector_cfg = detector_cfg
        self.max_detections = detector_cfg['TEST']['DETECTIONS_PER_IMAGE']
        self.fc_dim = detector_cfg['MODEL']['ROI_BOX_HEAD']['FC_DIM']
        self.num_classes = detector_cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']
        self.fc_layer = fc_layer

        self.detector_predictor = DefaultPredictor(detector_cfg)

        self.max_input_len = max_caption_length + self.max_detections + 3  # +3 for 2x[SEP] and [CLS]
        self.bert_cfg = BertConfig.from_json_file(bert_cfg_path)

        bert_state_dict = torch.load(bert_weights_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_cfg.bert_model)
        self.bert_cfg.vocab_size = len(self.tokenizer.vocab)
        mask_word_id, eos_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

        self.bert_decoder = BertForSeq2SeqDecoder.from_pretrained(pretrained_model_name=self.bert_cfg,
                                                                  state_dict=bert_state_dict,
                                                                  mask_word_id=mask_word_id,
                                                                  eos_id=eos_id).to(device)
        del bert_state_dict

        self.bert_decoder.eval()

    def __call__(self, img_path):

        with torch.no_grad():
            img = read_image(img_path, format="BGR")

            pred = self.detector_predictor(img)

            vis_feats, vis_pe = get_img_tensors(pred, fc_layer=self.fc_layer, fc_dim=self.fc_dim,
                                                num_classes=self.num_classes, max_detections=self.max_detections)

            input_ids, segment_ids, position_ids, attn_mask = \
                prepare_bert_caption_inf(self.tokenizer.convert_tokens_to_ids, vis_feats.shape[0],
                                         self.max_detections, self.max_input_len)

            vis_feats = vis_feats.unsqueeze(0).to(self.device)
            vis_pe = vis_pe.unsqueeze(0).to(self.device)
            input_ids = input_ids.unsqueeze(0).to(self.device)
            segment_ids = segment_ids.unsqueeze(0).to(self.device)
            position_ids = position_ids.unsqueeze(0).to(self.device)
            attn_mask = attn_mask.unsqueeze(0).to(self.device)

            traces = self.bert_decoder(vis_feats=vis_feats, vis_pe=vis_pe, input_ids=input_ids,
                                       token_type_ids=segment_ids, position_ids=position_ids, attention_mask=attn_mask,
                                       task_idx=self.bert_cfg.task_idx)

            output_ids = traces[0].tolist()

            for i in range(len(output_ids)):
                w_ids = output_ids[i]
                output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                output_tokens = []
                for t in output_buf:
                    if t in ("[SEP]", "[PAD]"):
                        break
                    output_tokens.append(t)

            return self.tokenizer.convert_tokens_to_string(output_tokens)
