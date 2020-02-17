import json
import os
import pickle as pkl
from random import randint, shuffle
from random import random as rand

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions

from detectron2.data.detection_utils import read_image


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


class PPCocoCaptions(Dataset):
    """
    Preprocessed Coco Captions, uses preprocessed data to increase GPU utilization
    """
    def __init__(self, data_dir):
        super().__init__()
        self._data_dir = data_dir
        # Hacky, fast way to get len because in the data_dir each caption has a file and a directory, thus // 2
        self._len = len(os.listdir(data_dir)) // 2

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        with open(os.path.join(self._data_dir, f'vis_feat_pe_{idx:06}.pkl'), 'rb') as vis_f:
            vis_feats = pkl.load(vis_f)
            vis_pe = pkl.load(vis_f)

        #vis_tensors = {'vis_feats': vis_feats, 'vis_pe': vis_pe}
        vis_tensors = [vis_feats, vis_pe]

        captions_tensors = []
        for j in range(5):
            with open(os.path.join(self._data_dir, f'vis_feat_pe_{idx:06}', f'caption_{j:02}.pkl'), 'rb') as cap_f:
                input_ids = pkl.load(cap_f)
                segment_ids = pkl.load(cap_f)
                attn_mask = pkl.load(cap_f)
                masked_ids = pkl.load(cap_f)
                masked_pos = pkl.load(cap_f)
                masked_weights = pkl.load(cap_f)

            #captions_tensors.append({'input_ids': input_ids, 'segment_ids': segment_ids, 'attn_mask': attn_mask,
            #                         'masked_ids': masked_ids, 'masked_pos': masked_pos,
            #                         'masked_weights': masked_weights})
            captions_tensors.append([input_ids, segment_ids, attn_mask, masked_ids, masked_pos, masked_weights])

        return vis_tensors, captions_tensors


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
