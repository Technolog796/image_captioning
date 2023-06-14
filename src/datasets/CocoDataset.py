import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset

import cv2
from PIL import Image

from transformers import AutoTokenizer
import open_clip

from tqdm.auto import tqdm
import json
import sys

import matplotlib.pyplot as plt


def read_image(path, size=(256, 256)):
    image = cv2.imread(path)
    image = cv2.resize(image, size)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


class CocoDataset(Dataset):
    def __init__(self, config, image_path="data/coco_dataset/train2014",
                 ann_path="data/coco_dataset/annotations/captions_train2014.json",
                 caption_path="data/coco_dataset/coco_train_trainslation.jsonl", data_type='train', coef_size=0.1,
                 tokenizer_name="",
                 prefix_length=20, normalize_prefix=False):
        if not tokenizer_name:
            tokenizer_name = config.decoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        _, _, self.preprocess = open_clip.create_model_and_transforms(config.encoder, pretrained="laion400m_e32")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        self.img_path = image_path
        self.ann_path = ann_path
        self.caption_path = caption_path
        coco_ds = dset.CocoDetection(root=image_path,
                                     annFile=ann_path)
        with open(caption_path, 'r') as f:
            q = list(f)[0]
            captions = json.loads(q)

        self.img_paths = []

        self.query_tokens = []
        self.answer_tokens = []
        max_seq_len = 0

        max_img = len(captions) * coef_size
        for i, caption_s in tqdm(enumerate(captions), total=max_img):
            for j in range(len(captions[caption_s])):
                query = "Что изображено на данной картинке?"
                answer = captions[caption_s][j]
                self.img_paths.append(caption_s)
                self.query_tokens.append(torch.tensor(self.tokenizer.encode(query), dtype=torch.int64))
                self.answer_tokens.append(torch.tensor(self.tokenizer.encode(answer), dtype=torch.int64))
                max_seq_len = max(max_seq_len, self.answer_tokens[-1].shape[0])
            if i >= max_img:
                break
        print("Data size is %0d" % len(self.img_paths))

        del captions
        del coco_ds
        sys.stdout.flush()
        all_len = torch.tensor([len(self.answer_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = prefix_length
        self.type = data_type

    def pad_tokens(self, item: int):
        query_tokens = self.query_tokens[item]
        padding = self.max_seq_len - query_tokens.shape[0]
        if padding > 0:
            query_tokens = torch.cat((query_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.query_tokens[item] = query_tokens
        elif padding < 0:
            query_tokens = query_tokens[:self.max_seq_len]
            self.query_tokens[item] = query_tokens
        query_mask = query_tokens.ge(0)
        query_tokens[~query_mask] = 0
        query_mask = query_mask.float()

        answer_tokens = self.answer_tokens[item]
        padding = self.max_seq_len - answer_tokens.shape[0]
        if padding > 0:
            answer_tokens = torch.cat((answer_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.answer_tokens[item] = answer_tokens
        elif padding < 0:
            answer_tokens = answer_tokens[:self.max_seq_len]
            self.answer_tokens[item] = answer_tokens
        answer_mask = answer_tokens.ge(0)
        answer_tokens[~answer_mask] = 0
        answer_mask = answer_mask.float()

        return query_tokens, query_mask, answer_tokens, answer_mask

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_image(self, item):
        name = str(self.img_paths[item])
        name = f"{self.img_path}/{name}"
        image = read_image(path=name)
        return image

    def __getitem__(self, item):
        image = self.get_image(item)
        image = self.preprocess(image).unsqueeze(0)
        query_tokens, query_mask, answer_tokens, answer_mask = self.pad_tokens(item)
        return query_tokens, query_mask, answer_tokens, answer_mask, image[0], item

    def show_image(self, item):
        img = self.get_image(item)
        text = self.tokenizer.decode(self.pad_tokens(item)[2])
        plt.imshow(img)
        print(text)
