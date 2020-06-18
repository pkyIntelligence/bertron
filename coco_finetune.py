from tqdm import tqdm
import argparse
from datetime import datetime
import logging
import os
import torch

from torch.utils.data import DataLoader
from torch.optim import *
from data_utils import PPCocoCaptions, prep_vis_pe, prepare_bert_caption_train

from VLP.pytorch_pretrained_bert.tokenization import BertTokenizer
from VLP.pytorch_pretrained_bert.modeling import BertConfig, BertForPreTrainingLossMask

"""
Example usage:

python coco_finetune.py \
    --model_path model_weights/bert/model.30.bin \
    --bert_config VLP/configs/bert_for_captioning.json \
    --gcp_bucket sc_data_pky \
    --gcp_auth_key auth/gcp_sbc_key.json \
    --batch_size 32 \
    --dl_workers 4
"""


def main():
    parser = argparse.ArgumentParser()

    # Model Setup
    parser.add_argument("--device_str", type=str, default="cuda:0",
                        help="The device which will do the training, usually cuda:0 or cpu")
    parser.add_argument("--model_path", type=str,
                        help="Path to the starting model state")
    parser.add_argument("--bert_config", type=str,
                        help="Path to bert configuration file")

    # GCP/Coco configuration
    parser.add_argument("--gcp_bucket", type=str,
                        help="The GCP bucket where to find the COCO data")
    parser.add_argument('--coco_root', type=str, default="coco",
                        help="The root to the COCO data in the GCP bucket")
    parser.add_argument('--gcp_auth_key', type=str,
                        help="Path to your GCP key which holds the COCO data")
    parser.add_argument("--num_detections", type=int, default=100,
                        help="The number of detections the detector was configured for")

    # For training details
    parser.add_argument('--feedback_rate', type=int, default=100,
                        help="This script will print the running loss every multiple of this many batches")
    parser.add_argument('--save_rate', type=int, default=500,
                        help="The model will save every multiple of this many batches")
    parser.add_argument('--epochs', type=int, default=1,
                        help="The number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for decoding. Highly recommended to be a multiple of 8")
    parser.add_argument('--dl_workers', type=int, default=3,
                        help="Number of dataloader workers")

    args = parser.parse_args()

    # Setting up logging
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    if not os.path.isdir("logging"):
        os.mkdir("logging")
    dt_format_string = "%Y-%m-%d_%H%M%S.%f"
    fh = logging.FileHandler(f"logging/{__file__}_{datetime.now().strftime(dt_format_string)}.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    device = torch.device(args.device_str)
    model_dir = '/'.join(args.model_path.split('/')[:-1])

    # max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
    state_dict = torch.load(args.model_path)
    bert_config = BertConfig.from_json_file(args.bert_config)
    tokenizer = BertTokenizer.from_pretrained(bert_config.bert_model)
    bert_config.vocab_size = len(tokenizer.vocab)

    model = BertForPreTrainingLossMask.from_pretrained(pretrained_model_name=bert_config,
                                                       state_dict=state_dict,
                                                       enable_butd=True,
                                                       len_vis_input=args.num_detections).to(device)

    model.train()
    logger.info("Bert Model loaded")

    train_ds = PPCocoCaptions(data_bucket=args.gcp_bucket,
                              dataset_root=args.coco_root,
                              auth_key_file=args.gcp_auth_key)

    pin_memory = (args.device_str == "cuda:0")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.dl_workers, pin_memory=pin_memory)

    # same parameters as BERT
    optimizer = Adam(params=model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch} started")
        # Each image has 5 sentences, each sentence is an example
        running_loss = 0

        for i, data in enumerate(tqdm(train_dl), 0):
            # get the inputs; data is a list of [imgs, captions]

            feats, bbox_preds, cls_probs, sents = data

            feats = feats.to(device)
            bbox_preds = bbox_preds.to(device)
            cls_probs = cls_probs.to(device)

            vis_pe = prep_vis_pe(bbox_preds, cls_probs)

            input_ids_list = []
            segment_ids_list = []
            attn_mask_list = []
            masked_ids_list = []
            masked_pos_list = []
            masked_weights_list = []

            for sent in sents['raw']:
                input_ids, segment_ids, attn_mask, masked_ids, masked_pos, masked_weights = \
                    prepare_bert_caption_train(tokenizer, num_detections=args.num_detections, caption=sent)
                input_ids_list.append(input_ids)
                segment_ids_list.append(segment_ids)
                attn_mask_list.append(attn_mask)
                masked_ids_list.append(masked_ids)
                masked_pos_list.append(masked_pos)
                masked_weights_list.append(masked_weights)

            batched_input_ids = torch.stack(input_ids_list).to(device)
            batched_segment_ids = torch.stack(segment_ids_list).to(device)
            batched_attn_mask = torch.stack(attn_mask_list).to(device)
            batched_masked_ids = torch.stack(masked_ids_list).to(device)
            batched_masked_pos = torch.stack(masked_pos_list).to(device)
            batched_masked_weights = torch.stack(masked_weights_list).to(device)

            loss_tuple = model(vis_feats=feats, vis_pe=vis_pe, input_ids=batched_input_ids,
                               token_type_ids=batched_segment_ids, attention_mask=batched_attn_mask,
                               masked_lm_labels=batched_masked_ids, next_sentence_label=-1,
                               masked_pos=batched_masked_pos, masked_weights=batched_masked_weights,
                               task_idx=3, drop_worst_ratio=0.2)
            masked_lm_loss, pretext_loss_deprecated, ans_loss = loss_tuple
            loss = masked_lm_loss + pretext_loss_deprecated + ans_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print feedback
            running_loss += loss.item()
            if i % args.feedback_rate == (args.feedback_rate-1):
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / args.feedback_rate}')
                running_loss = 0.0

            if i % args.save_rate == (args.save_rate-1):
                logger.info(f"Saving Model in epoch {epoch} and iteration {i}")
                torch.save(model.state_dict(), os.path.join(model_dir, f"coco_bert_{(i+1):03}.pt"))

    torch.save(model.state_dict(), os.path.join(model_dir, "coco_bert_final.pt"))


if __name__ == "__main__":
    main()
