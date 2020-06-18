from torch.utils.data.dataloader import DataLoader

import argparse
import math
import numpy as np
import random
from tqdm import tqdm

from VLP.vlp.lang_utils import language_eval

from data_utils import *
from captioner import Captioner


"""
Example usage:

python validate_coco_captions.py \
    --detector_config detectron2/configs/COCO-Detection/faster_rcnn_X_101_64x4d_FPN_2x_vlp.yaml \
    --detector_weights model_weights/detectron/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp.pkl \
    --decoder_config VLP/configs/bert_for_captioning.json \
    --decoder_weights model_weights/bert/model.19.bin \
    --object_vocab vocab/objects_vocab.txt \
    --coco_root ~/Datasets/coco \
    --coco_data_info annotations/dataset_coco.json \
    --coco_ann_file annotations/captions_val2014.json \
    --valid_jpgs_file annotations/coco_valid_jpgs.json \
    --batch_size 4 \
    --dl_workers 4
"""


def main():
    parser = argparse.ArgumentParser()

    # Model Setup
    parser.add_argument("--detector_config", default=None, type=str,
                        help="detector config file path.")
    parser.add_argument("--detector_weights", default=None, type=str,
                        help="pretrained detector weights.")
    parser.add_argument("--decoder_config", default=None, type=str,
                        help="Bert decoder config file path.")
    parser.add_argument("--decoder_weights", default=None, type=str,
                        help="pretrained Bert decoder weights.")
    parser.add_argument("--object_vocab", default=None, type=str,
                        help="object vocabulary, maps object ids to object names")

    # For COCO
    parser.add_argument('--coco_root', type=str, default='~/Datasets/coco')
    parser.add_argument("--coco_data_info", default='annotations/dataset_coco.json', type=str,
                        help="The input data file name.")
    parser.add_argument("--coco_ann_file", default='annotations/captions_val2014.json', type=str,
                        help="caption annotations file (i.e. answer key)")
    parser.add_argument('--valid_jpgs_file', default='annotations/coco_valid_jpgs.json', type=str,
                        help="lists the valid jpgs")

    # For data pipeline
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for decoding. Highly recommended to be a multiple of 8")
    parser.add_argument('--dl_workers', type=int, default=0, help="Number of dataloader workers")

    # For reproducibility
    parser.add_argument('--seed', type=int, default=-1, help="random seed for initialization")

    args = parser.parse_args()

    assert(torch.cuda.is_available())

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()

    # fix random seed (optional)
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():

        captioner = Captioner(args.detector_config, args.detector_weights, args.decoder_config, args.decoder_weights,
                              args.object_vocab, cpu_device, gpu_device)

        # TODO: optimize for amp, data-parallel
        torch.cuda.empty_cache()  # Empty everything

        valid_dataset = CocoCaptionsKarpathyValidImgs(args.coco_root)
        valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=ccc_karpathy_valid_collate,
                              num_workers=args.dl_workers, pin_memory=True)

        total_batch = math.ceil(len(valid_dataset) / args.batch_size)

        predictions = []

        print('start the caption evaluation...')
        with tqdm(total=total_batch) as pbar:
            for img_ids, img_npys in valid_dl:
                captions = captioner.forward(img_npys)

                for img_id, caption in zip(img_ids, captions):
                    predictions.append({'image_id': img_id, 'caption': caption})
                pbar.update(1)

        language_eval(preds=predictions, annFile=os.path.join(args.coco_root, args.coco_ann_file),
                      model_id='0', split='val')


if __name__ == "__main__":
    main()
