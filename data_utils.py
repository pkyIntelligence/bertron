import json
import matplotlib.pyplot as plt
import os
import pickle as pkl
import h5py
from random import randint, shuffle
from random import random as rand

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
from torchvision.datasets import CocoCaptions

from detectron2.data.detection_utils import read_image

from gcp_file_handler import GCPFileHandler


class CocoCaptionsKarpathyValidImgs(Dataset):
    """
    A Dataset which represents the valid jpgs for Karpathy's split on COCO
    """
    def __init__(self, data_dir):
        """
        args:
            data_dir: the root to where coco images are stored
        """
        super().__init__()
        self._data_dir = data_dir
        self.eval_list = []
        with open(os.path.join(data_dir, 'annotations/dataset_coco.json'), "r", encoding='utf-8') as f_src:
            img_dat = json.load(f_src)['images']
            valid_jpgs = json.load(open(os.path.join(data_dir, 'annotations/coco_valid_jpgs.json')))
            for src in img_dat:
                if src['split'] == 'val' and (valid_jpgs is None or src['filename'] in valid_jpgs):
                    src_tk = os.path.join(data_dir, src.get('filepath', 'trainval'), src['filename'])
                    imgid = int(src['filename'].split('_')[2][:-4])
                    self.eval_list.append((imgid, src_tk))  # id and path for COCO

        self._len = len(self.eval_list)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img_info = self.eval_list[idx]
        img_npy = read_image(img_info[1], format="BGR")

        return img_info[0], img_npy


class DebugCocoCaptionsKarpathyValidImgs(CocoCaptionsKarpathyValidImgs):
    """
    modifying some methods to allow easier debugging, but not necessarily very performant
    """
    def __getitem__(self, slice_obj):
        if type(slice_obj) == int:
            return super().__getitem__(slice_obj)
        stop = slice_obj.stop
        start = 0 if slice_obj.start is None else slice_obj.start
        step = 1 if slice_obj.step is None else slice_obj.step
        out = []
        for i in range(start, stop, step):
            out.append(super().__getitem__(i))

        return out


def ccc_karpathy_valid_collate(batch):
    return list(zip(*batch))


class PPCocoCaptions(IterableDataset):
    """
    Preprocessed Coco Captions, uses preprocessed data from the cloud
    """
    def __init__(self, data_bucket, dataset_root, auth_key_file):
        super().__init__()
        self.data_bucket = data_bucket
        self.dataset_root = dataset_root
        self.auth_key_file = auth_key_file
        # create annotations indexed by the final 3 numbers of the filename as preprocessed features are arranged in
        # the cloud files that way
        self.anns = {f"{i:03}": [] for i in range(1000)}
        with GCPFileHandler(bucket_name=data_bucket,
                            source_blob_name=dataset_root + "/annotations/dataset_coco.json",
                            destination_file_name="tmp_annotations.json",
                            auth_key_file=auth_key_file) as gcp_ann_file, \
            GCPFileHandler(bucket_name=data_bucket,
                            source_blob_name=dataset_root + "/annotations/coco_valid_jpgs.json",
                            destination_file_name="tmp_valid_jpgs.json",
                            auth_key_file=auth_key_file) as gcp_valid_file:
            with open(gcp_ann_file.filename, 'r', encoding='utf-8') as ann_file, \
                open(gcp_valid_file.filename, 'r', encoding='utf-8') as valid_file:

                ann_list = json.load(ann_file)['images']
                valid_jpgs = json.load(valid_file)

                for ann in ann_list:
                    if ann['filename'] in valid_jpgs.keys():
                        self.anns[ann['filename'].split('.')[0][-3:]].append(ann)

        self.bbox_file = GCPFileHandler(bucket_name=self.data_bucket,
                            source_blob_name=self.dataset_root + "/region_feat_gvd_wo_bgd/coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5",
                            destination_file_name="tmp_bbox.h5",
                            auth_key_file=self.auth_key_file)
        self.bbox_filename = "tmp_bbox.h5"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        my_idxs = range(worker_info.id, 1000, worker_info.num_workers)
        for idx in my_idxs:
            key = f"{idx:03}"
            ann_list = self.anns[key]
            with GCPFileHandler(bucket_name=self.data_bucket,
                                source_blob_name=self.dataset_root + f"/region_feat_gvd_wo_bgd/feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval_feat{key}.h5",
                                destination_file_name=f"tmp_feat{key}.h5",
                                auth_key_file=self.auth_key_file) as feat_file, \
                 GCPFileHandler(bucket_name=self.data_bucket,
                                source_blob_name=self.dataset_root + f"/region_feat_gvd_wo_bgd/feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval_cls{key}.h5",
                                destination_file_name=f"tmp_cls{key}.h5",
                                auth_key_file=self.auth_key_file) as cls_file:
                with h5py.File(feat_file.filename, 'r') as feat_h5_file, \
                        h5py.File(cls_file.filename, 'r') as cls_h5_file, \
                        h5py.File(self.bbox_filename, 'r') as bbox_h5_file:
                    for ann in ann_list:
                        img_name = ann['filename'].split('.')[0]
                        feats = feat_h5_file[img_name][:]
                        cls_probs = cls_h5_file[img_name][:]
                        bbox_preds = bbox_h5_file[img_name][:]

                        for sent in ann['sentences']:
                            yield feats, bbox_preds, cls_probs, sent

    def __del__(self):
        self.bbox_h5_file.close()


class DebugCocoCaptions(CocoCaptions):
    """
    modifying some methods to allow easier debugging, but not necessarily very performant
    """
    def __getitem__(self, slice_obj):
        if type(slice_obj) == int:
            return super().__getitem__(slice_obj)
        stop = slice_obj.stop
        start = 0 if slice_obj.start is None else slice_obj.start
        step = 1 if slice_obj.step is None else slice_obj.step
        out = []
        for i in range(start, stop, step):
            out.append(super().__getitem__(i))

        return out


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def get_img_tensors(preds, fc_layer, fc_dim, num_classes, max_detections=100):
    """
    Args:
        preds: predictions from a detectron2 detector, a list of instances
        fc_layer: 0-indexed layer to pull features from (in prior literature FC6 = 0 (first FC layer), and
                    FC7 = 1 (2nd FC layer))
        fc_dim:

    Returns:
        box_features: tensor of box features from the FC layer output, shape = (number of regions, fc-dim)
        vis_pe: visual positional embedding, which is bbox + area + box score
        both are end row padded to the max detection limit
    """
    h, w = preds['instances'].image_size
    fields = preds['instances'].get_fields()

    fc_box_features = fields['fc_box_features'][:, fc_layer*fc_dim:(fc_layer+1)*fc_dim]
    probs = fields['probs']
    boxes = fields['pred_boxes'].clone()

    num_detections = fc_box_features.shape[0]

    boxes.scale(scale_x=1/w, scale_y=1/h)
    areas = boxes.area().unsqueeze(dim=1)
    scores = fields['scores'].unsqueeze(dim=1)

    bbox_areas = torch.cat([boxes.tensor, areas, scores], dim=1)
    #  4 coordinates + 1 bbox area +1 score, +1 for background class
    vis_pe = torch.cat((F.layer_norm(bbox_areas, [6]), F.layer_norm(probs, [num_classes+1])), dim=-1)

    box_features = F.pad(fc_box_features, [0, 0, 0, max_detections-num_detections])
    vis_pe = F.pad(vis_pe, [0, 0, 0, max_detections-num_detections])

    return box_features, vis_pe


def prep_vis_pe(bbox_preds, cls_probs):
    """
    Args:
        bbox_preds: raw pre-processed bbox predictions from detector, shape = (batch, detections, 6)
        cls_probs: raw pre-processed class probabilities from detector, shape = (batch, detections, num classes + 1)

    Returns:
        vis_pe: visual positional embedding, which is norm bbox + norm area + box score
            shape = (batch, detections, 6)
    """
    batch_size = bbox_preds.shape[0]
    num_detections = bbox_preds.shape[1]
    max_x1s, _ = torch.max(bbox_preds[:, :, 0], dim=1)
    max_x2s, _ = torch.max(bbox_preds[:, :, 2], dim=1)
    max_y1s, _ = torch.max(bbox_preds[:, :, 1], dim=1)
    max_y2s, _ = torch.max(bbox_preds[:, :, 3], dim=1)
    w_ests = torch.max(max_x1s, max_x2s)*1.+1e-5
    h_ests = torch.max(max_y1s, max_y2s)*1.+1e-5
    bbox_preds[:, :, [0, 2]] = torch.div(bbox_preds[:, :, [0, 2]], w_ests.unsqueeze(1).unsqueeze(2))
    bbox_preds[:, :, [1, 3]] = torch.div(bbox_preds[:, :, [1, 3]], h_ests.unsqueeze(1).unsqueeze(2))

    rel_area = (bbox_preds[:, :, 3]-bbox_preds[:, :, 1])*(bbox_preds[:, :, 2]-bbox_preds[:, :, 0])
    rel_area.clamp_(0)

    vis_pe = torch.cat((bbox_preds[:, :, :4],
                        rel_area.view(batch_size, num_detections, 1),
                        bbox_preds[:, :, 5:]), dim=-1)
    vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), F.layer_norm(cls_probs, [1601])), dim=-1)

    return vis_pe

def prepare_bert_caption_train(tokenizer, num_detections, caption, max_input_len=170, max_n_mask=10,
                               mask_prob=0.15):
    """
    Args:
        tokenizer: tokenizer which tokenizes the caption
        num_detections: number of visual regions for this input
        caption: a text string to be masked
        device: device to create the tensors in
        max_input_len: the max sequence length for the entire input including [CLS] and 2x [SEP]
        max_n_mask: the max number of [MASK] tokens possible, used to ensure tensor is not ragged
        mask_prob: probability to mask a token for training

    Returns:
        input_ids: Padded tokens to be filled in during embedding layer
        segment_ids: [4] for image, [5] for caption masks and subsequent inference
        position_ids: for positional embedding calculation
        attention_mask: attention mask for inference, all visual features can attend bi-directionally to themselves but
            not the caption, captioning can attend to all visual features and earlier in sequence but not the future

    """
    tokens_a = ['[UNK]'] * num_detections
    tokens_b = tokenizer.tokenize(caption)

    # Add Special Tokens
    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

    # 4: visual features segment id, 5: caption segment id
    segment_ids = [4]*(len(tokens_a)+2) + [5]*(len(tokens_b)+1)

    # Random Masking
    n_masks = round(mask_prob * len(tokens_b))
    # candidate positions of masked tokens
    cand_pos = list(range(len(tokens_a)+2, len(tokens)))
    shuffle(cand_pos)
    masked_pos = cand_pos[:n_masks]

    # Removed vis_masked_pos for now

    masked_tokens = [tokens[pos] for pos in masked_pos]
    for pos in masked_pos:
        if rand() < 0.8:  # 80%, replace with MASK token
            tokens[pos] = '[MASK]'
        elif rand() < 0.5:  # 50% of (1-80%) = 10%, random word
            tokens[pos] = get_random_word(list(tokenizer.vocab.keys()))

    masked_weights = [1]*len(masked_tokens)

    # Token Indexing
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # [[CLS], [UNK]*vis_input_len, [SEP], tokens_b [SEP]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Zero Padding
    n_pad = max_input_len - len(input_ids)
    input_ids.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)

    # Zero Padding for Mask targets
    n_pad = max_n_mask - n_masks
    masked_ids.extend([0]*n_pad)
    masked_pos.extend([0]*n_pad)
    masked_weights.extend([0]*n_pad)

    # Convert to tensors
    masked_pos = torch.LongTensor(masked_pos)
    masked_ids = torch.LongTensor(masked_ids)
    masked_weights = torch.LongTensor(masked_weights)
    input_ids = torch.LongTensor(input_ids)
    segment_ids = torch.LongTensor(segment_ids)

    # creating attention mask
    attn_mask = torch.zeros(max_input_len, max_input_len, dtype=torch.long)
    attn_mask[:, :len(tokens_a)+2].fill_(1)  # 1's on the visual input but not the padding
    b_start, b_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3
    attn_mask[b_start:b_end, b_start:b_end] = torch.tril(torch.ones(b_end-b_start, b_end-b_start, dtype=torch.long))

    return input_ids, segment_ids, attn_mask, masked_ids, masked_pos, masked_weights


def prepare_bert_caption_inf(indexer, num_detections, max_detections=100, max_input_len=170):
    """
    Args:
        indexer: converts the tokens to id's by the tokenizer
        num_detections: number of visual regions for this input
        max_detections: the max number of regions that can be detected
        max_input_len: the max sequence length for the entire input including [CLS] and 2x [SEP]

    Returns:
        input_ids: Padded tokens to be filled in during embedding layer
        segment_ids: [4] for image, [5] for caption masks and subsequent inference
        position_ids: for positional embedding calculation
        attention_mask: attention mask for inference, all visual features can attend bi-directionally to themselves but
            not the caption, captioning can attend to all visual features and earlier in sequence but not the future

    """
    tokens_a = ['[UNK]'] * num_detections

    # Add Special Tokens
    padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
    assert len(padded_tokens_a) <= max_detections + 2
    if max_detections + 2 > len(padded_tokens_a):  # need to pad
        padded_tokens_a += ['[PAD]'] * (max_detections + 2 - len(padded_tokens_a))  # pad to max detections
    assert len(padded_tokens_a) == max_detections + 2

    tokens = padded_tokens_a

    segment_ids = torch.LongTensor([4] * (len(padded_tokens_a)) + [5] * (max_input_len - len(padded_tokens_a)))

    position_ids = []
    for i in range(len(tokens_a) + 2):
        position_ids.append(i)
    for i in range(len(tokens_a) + 2, max_detections + 2):
        position_ids.append(0)
    for i in range(max_detections + 2, max_input_len):
        position_ids.append(i - (max_detections + 2) + len(tokens_a) + 2)
    position_ids = torch.LongTensor(position_ids)

    # Token Indexing
    input_ids = torch.LongTensor(indexer(tokens))  # [[CLS], [UNK]*vis_input_len, [SEP], [PAD]*rest]

    # creating attention mask
    attn_mask = torch.zeros(max_input_len, max_input_len, dtype=torch.long)
    attn_mask[:, :len(tokens_a)+2].fill_(1)  # 1's on the visual input but not the padding
    b_start, b_end = len(padded_tokens_a), max_input_len
    attn_mask[b_start:b_end, b_start:b_end] = torch.tril(torch.ones(b_end-b_start, b_end-b_start, dtype=torch.long))

    return input_ids, segment_ids, position_ids, attn_mask

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')