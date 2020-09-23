import validators
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultMultiImgPredictor
from detectron2.utils.visualizer import ColorMode, SimpleVisualizer

from VLP.pytorch_pretrained_bert.modeling import BertConfig, BertForSeq2SeqDecoder
from VLP.pytorch_pretrained_bert.tokenization import BertTokenizer

from data_utils import *

# Hack to load from python 2.7 models
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

class Captioner:
    """
    A Captioner, some sort of detector or image feature extractor combined with a decoder to generate the caption
    """
    def __init__(self, detector_cfg_path, detector_weights_path, bert_cfg_path, bert_weights_path, object_vocab_path,
                 cpu_device, gpu_device, fc_layer=0, max_caption_length=67):
        """
        args:
            detector_cfg_path: path to the detector config
            detector_weights_path: path to the detector weights
            bert_cfg_path: path to the bert decoder config
            bert_weights_path: path to the bert decoder weights
            cpu_device: The cpu device to run some parts of visualization
            gpu_device: The gpu device to run the bulk of computations, currently requires at least 1 GPU device
            fc_layer: the fully connected layer from the detector to extract features from, 0-indexed
            max_caption_length: the maximum number of tokens the caption can be
        """
        self.cpu_device = cpu_device
        self.gpu_device = gpu_device

        detector_cfg = get_cfg()
        detector_cfg.merge_from_file(detector_cfg_path)
        detector_cfg.merge_from_list(['MODEL.WEIGHTS', detector_weights_path])
        detector_cfg.freeze()

        self.detector_cfg = detector_cfg
        self.max_detections = detector_cfg['TEST']['DETECTIONS_PER_IMAGE']
        self.fc_dim = detector_cfg['MODEL']['ROI_BOX_HEAD']['FC_DIM']
        self.num_classes = detector_cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']
        self.fc_layer = fc_layer
        self.metadata = MetadataCatalog.get(
            detector_cfg.DATASETS.TEST[0] if len(detector_cfg.DATASETS.TEST) else "__unused"
        )

        self.detector_predictor = DefaultMultiImgPredictor(detector_cfg)

        self.max_input_len = max_caption_length + self.max_detections + 3  # +3 for 2x[SEP] and [CLS]
        self.bert_cfg = BertConfig.from_json_file(bert_cfg_path)

        bert_state_dict = torch.load(bert_weights_path, pickle_module=pickle)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_cfg.bert_model)
        self.bert_cfg.vocab_size = len(self.tokenizer.vocab)
        mask_word_id, eos_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

        self.bert_decoder = BertForSeq2SeqDecoder.from_pretrained(pretrained_model_name=self.bert_cfg,
                                                                  state_dict=bert_state_dict,
                                                                  mask_word_id=mask_word_id,
                                                                  eos_id=eos_id)

        if gpu_device:
            self.bert_decoder.to(gpu_device)
        else:
            self.bert_decoder.to(cpu_device)
        del bert_state_dict

        self.bert_decoder.eval()

        with open(object_vocab_path, 'r') as f:
            self.obj_class_names = f.read().splitlines()

        self.img_cache = None
        self.vis_pred_cache = None

    def visualize(self, img=None, top_n=100):
        """
        Visualize the detector output.

        args:
            img: a path or url to an image file or image object, if None, assume intent is to visualize the last run
            top_n: only show this many top scoring detections, to make it cleaner, captioning still uses top 100.
        """
        if img is None:
            img = self.img_cache
            pred = self.vis_pred_cache
        elif isinstance(img, str):
            if validators.url(img):
                response = requests.get(img)
                img = np.array(Image.open(BytesIO(response.content)))[:, :, ::-1]
            else:  # Assume it's a file path
                img = read_image(img, format="BGR")
            pred = self.detector_predictor([img])[0]
        else:  # Assume it's an image object
            pred = self.detector_predictor([img])[0]

        self.img_cache = img
        self.vis_pred_cache = pred

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        img = img[:, :, ::-1]
        visualizer = SimpleVisualizer(img, self.metadata, self.obj_class_names, instance_mode=ColorMode.IMAGE)

        instances = pred["instances"][:top_n].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances).get_image()

        return vis_output

    def __call__(self, img, visualize=False, viz_top_n=100):
        """
        inference only for now
        args:
            img: path or url to the image to caption or img array
            visualize: Do you want to show the detector output? (makes things slightly slower)
            viz_top_n: if visualize is True, this is how many top scoring detections will be visualized, otherwise
                        it is ignored, captioning will still use top 100 regardless of this value.
        """

        with torch.no_grad():
            if isinstance(img, str):
                if validators.url(img):
                    response = requests.get(img)
                    img = np.array(Image.open(BytesIO(response.content)))[:, :, ::-1]
                else:  # Assume file path
                    img = read_image(img, format="BGR")

            pred = self.detector_predictor([img])[0]
            self.img_cache = img
            self.vis_pred_cache = pred

            if visualize:
                vis_output = self.visualize(img, viz_top_n)
            else:
                vis_output = img[:, :, ::-1]

            vis_feats, vis_pe = get_img_tensors(pred, fc_layer=self.fc_layer, fc_dim=self.fc_dim,
                                                num_classes=self.num_classes, max_detections=self.max_detections)

            input_ids, segment_ids, position_ids, attn_mask = \
                prepare_bert_caption_inf(self.tokenizer.convert_tokens_to_ids, vis_feats.shape[0],
                                         self.max_detections, self.max_input_len)

            vis_feats = vis_feats.unsqueeze(0)
            vis_pe = vis_pe.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)
            segment_ids = segment_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(0)

            if self.gpu_device:
                vis_feats = vis_feats.to(self.gpu_device)
                vis_pe = vis_pe.to(self.gpu_device)
                input_ids = input_ids.to(self.gpu_device)
                segment_ids = segment_ids.to(self.gpu_device)
                position_ids = position_ids.to(self.gpu_device)
                attn_mask = attn_mask.to(self.gpu_device)

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

            return self.tokenizer.convert_tokens_to_string(output_tokens), vis_output

    def forward(self, img_npys):
        """
        process image captions in batches
        args:
            img_npys: a list of img numpy arrays
        """
        preds = self.detector_predictor(img_npys)

        vis_feats_list = []
        vis_pe_list = []
        input_ids_list = []
        segment_ids_list = []
        position_ids_list = []
        attn_mask_list = []

        for pred in preds:
            vis_feats, vis_pe = get_img_tensors(pred, fc_layer=self.fc_layer, fc_dim=self.fc_dim,
                                                num_classes=self.num_classes, max_detections=self.max_detections)

            input_ids, segment_ids, position_ids, attn_mask = \
                prepare_bert_caption_inf(self.tokenizer.convert_tokens_to_ids, vis_feats.shape[0],
                                         self.max_detections, self.max_input_len)

            device = self.gpu_device if self.gpu_device else self.cpu_device

            vis_feats_list.append(vis_feats.unsqueeze(0).to(device))
            vis_pe_list.append(vis_pe.unsqueeze(0).to(device))
            input_ids_list.append(input_ids.unsqueeze(0).to(device))
            segment_ids_list.append(segment_ids.unsqueeze(0).to(device))
            position_ids_list.append(position_ids.unsqueeze(0).to(device))
            attn_mask_list.append(attn_mask.unsqueeze(0).to(device))

        batch_vis_feats = torch.cat(vis_feats_list)
        batch_vis_pe = torch.cat(vis_pe_list)
        batch_input_ids = torch.cat(input_ids_list)
        batch_segment_ids = torch.cat(segment_ids_list)
        batch_position_ids = torch.cat(position_ids_list)
        batch_attn_mask = torch.cat(attn_mask_list)

        traces = self.bert_decoder(vis_feats=batch_vis_feats, vis_pe=batch_vis_pe, input_ids=batch_input_ids,
                                   token_type_ids=batch_segment_ids, position_ids=batch_position_ids,
                                   attention_mask=batch_attn_mask, task_idx=self.bert_cfg.task_idx)

        batch_tokens_tensor = traces[0]
        batch_size = batch_tokens_tensor.shape[0]

        output_tokens_list = []
        for i in range(batch_size):
            token_ids = batch_tokens_tensor[i, :].tolist()
            output_buf = self.tokenizer.convert_ids_to_tokens(token_ids)
            output_tokens = []
            for t in output_buf:
                if t in ("[SEP]", "[PAD]"):
                    break
                output_tokens.append(t)

            output_tokens_list.append(output_tokens)

        return [self.tokenizer.convert_tokens_to_string(output_tokens) for output_tokens in output_tokens_list]