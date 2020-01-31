from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import *

from transformers import *
from transformers.modeling_bert import BertForPreTrainingLossMask

from Capstone.data_utils import *


MAX_TGT_LENGTH = 67
DEVICE = torch.device('cuda:0')
DATA_DIR = r'D:\datasets\coco2014_features_cap'
MODEL_DIR = r'D:\models'
MASK_PROB = 0.15

# max_input_len = MAX_TGT_LENGTH + max_detections + 3  # +3 for 2x[SEP] and [CLS]
state_dict = torch.load(os.path.join('Capstone', 'model_weights', 'model.30.bin'))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])

model = BertForPreTrainingLossMask.from_pretrained(
            pretrained_model_name_or_path='bert-base-cased',
            config=None,
            state_dict=state_dict,
            num_labels=2,
            len_vis_input=100,
            type_vocab_size=6,
        ).to(DEVICE)
model.train()

train_ds = PPCocoCaptions(DATA_DIR)
train_dl = DataLoader(train_ds, batch_size=16, num_workers=4, pin_memory=False)

# same parameters as BERT
optimizer = Adam(params=model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
optimizer.zero_grad()  # probably unnecessary?


# Each epoch is kind of like 5 epochs with 5 captions
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dl), 0):
        # get the inputs; data is a list of [imgs, captions]

        vis_tensors, captions_tensors = data
        vis_tensors[0] = vis_tensors[0].to(DEVICE)
        vis_tensors[1] = vis_tensors[1].to(DEVICE)

        for caption_tensors in captions_tensors:

            caption_tensors[0] = caption_tensors[0].to(DEVICE)
            caption_tensors[1] = caption_tensors[1].to(DEVICE)
            caption_tensors[2] = caption_tensors[2].to(DEVICE)
            caption_tensors[3] = caption_tensors[3].to(DEVICE)
            caption_tensors[4] = caption_tensors[4].to(DEVICE)
            caption_tensors[5] = caption_tensors[5].to(DEVICE)

            masked_lm_loss = model(vis_feats=vis_tensors[0], vis_pe=vis_tensors[1], input_ids=caption_tensors[0],
                                   token_type_ids=caption_tensors[1], attention_mask=caption_tensors[2],
                                   masked_lm_labels=caption_tensors[3], masked_pos=caption_tensors[4],
                                   masked_weights=caption_tensors[5], task_idx=3, drop_worst_ratio=0.2)

            masked_lm_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print feedback
        running_loss += masked_lm_loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss/10}')
            running_loss = 0.0

torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'coco_caption_bert.pt'))
