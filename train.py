import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import math
from einops import rearrange
from torch import einsum
from torch.optim import AdamW
import random

from torch.nn import functional as nnf

from tqdm.auto import tqdm
from tqdm.contrib import tzip

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertConfig, BertLMHeadModel, Blip2QFormerModel
from transformers import AutoTokenizer
import json
import wandb
import sys
import cv2
import torchvision.datasets as dset
from PIL import Image
import pickle
import os

from typing import Tuple, Optional, Union, Any
from torch.cuda.amp import autocast
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
import pandas as pd
import open_clip
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

from torchmetrics import BLEUScore
from evaluate import load
from statistics import mean
from torch.optim.lr_scheduler import CosineAnnealingLR

import re
import string

from src.config import Config
from src.datasets.CocoDataset import CocoDataset
from src.models.Model import ClipCaptionModel

import wandb

def truncate_sentences(sentences):
    truncated_sentences = []
    exclude = set(string.punctuation)
    for sentence in sentences:
        truncated_sentence = sentence
        index = truncated_sentence.find("<pad>")
        if index != -1:
            truncated_sentence = truncated_sentence[:index]
        truncated_sentence = ''.join(ch for ch in truncated_sentence if ch not in exclude)
        index = truncated_sentence.find("бродить")
        if index != -1:
            truncated_sentence = truncated_sentence[index + 8:]
        truncated_sentences.append(truncated_sentence)
    return truncated_sentences

bertscore = load("bertscore")
meteor = load('meteor')
rouge = load('rouge')
bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3]] + [bertscore, meteor, rouge]

def train(model, optimizer, scheduler, loss_func, loader, epoch, args):
    model.train()
    pbar = tqdm(loader, total=len(loader))
    step = 0
    for (query_tokens, query_mask, answer_tokens, answer_mask, prefix, idx) in pbar:

        query_tokens, query_mask, prefix = query_tokens.to(args.device), query_mask.to(args.device), prefix.to(
            args.device, dtype=torch.bfloat16)
        answer_tokens, answer_mask = answer_tokens.to(args.device), answer_mask.to(args.device)
        outputs, proj = model(query_tokens, query_mask, answer_tokens, answer_mask, prefix)
        logits = outputs.logits
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_tokens.flatten().to(torch.int64),
                                 ignore_index=0)

        loss2 = model.dist_loss(model.gpt.transformer.wte(answer_tokens).to(torch.float32), proj.to(torch.float32))
        loss += loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item(), "dist_loss": loss2.item()})
        wandb.log({"loss": loss.item(), "dist_loss": loss2.item()})
        step += 1
        if step % 1000 == 0:
            print("TEXT:", train_dataset.tokenizer.decode(answer_tokens[0]))
            print("PREDICTED: ", model.generate(torch.tensor([train_dataset[idx[0]][4].tolist()]).to(args.device),
                                                ["Что изображено на данной картинке?"])[0])
    with open(f'{args.save_path}checkpoint_{epoch}.pkl', 'wb') as f:
        pickle.dump(model, f)

def evaluate(model, optimizer, scheduler, loss_func, loader, args):
    model.eval()
    pbar = tqdm(loader, total=len(loader))
    step = 0

    bl1 = []
    bl2 = []
    bl3 = []
    brt = []
    mtr = []
    rg = []
    val_losses = []
    val_dist = []
    for (query_tokens, query_mask, answer_tokens, answer_mask, prefix, idx) in pbar:
        query_tokens, query_mask, prefix = query_tokens.to(args.device), query_mask.to(args.device), prefix.to(
            args.device, dtype=torch.bfloat16)
        answer_tokens, answer_mask = answer_tokens.to(args.device), answer_mask.to(args.device)
        outputs, proj = model(query_tokens, query_mask, answer_tokens, answer_mask, prefix)
        logits = outputs.logits
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_tokens.flatten().to(torch.int64),
                                 ignore_index=0)
        loss2 = model.dist_loss(model.gpt.transformer.wte(answer_tokens), proj)

        real = model.tokenizer.batch_decode(answer_tokens)
        pred = model.generate(torch.tensor([val_dataset[idx[j]][4].tolist() for j in range(len(idx))]).to(args.device),
                              ["Что изображено на данной картинке? " for _ in range(len(idx))])

        real = truncate_sentences(real)
        pred = truncate_sentences(pred)

        bl1.append(bleu_scorers[0](pred, real))
        bl2.append(bleu_scorers[1](pred, real))
        bl3.append(bleu_scorers[2](pred, real))
        brt.append(bleu_scorers[3].compute(predictions=pred, references=real, lang="ru")['f1'])
        mtr.append(bleu_scorers[4].compute(predictions=pred, references=real)['meteor'])
        rg.append(bleu_scorers[5].compute(predictions=pred, references=real)['rougeL'])

        if step % 400 == 0:
            print("TEXT:", real[0])
            print("PREDICTED: ", pred[0])

            imgs = []
            for j in range(len(idx)):
                wa_img = wandb.Image(
                    val_dataset.get_image(idx[j]),
                    caption=f"REAL : {real[j]}, PREDICTED : {pred[j]}"
                )
                imgs.append(wa_img)

            wandb.log({"Generations.": imgs})

        step += 1

        pbar.set_postfix({"val_loss": loss.item(), "val_dist": loss2.item()})
        val_losses.append(loss.item())
        val_dist.append(loss2.item())

    wandb.log({"val_loss": mean(val_losses),
               "val_dist": mean(val_dist)})

    wandb.log({
        "blue_1": mean([tensor.item() for tensor in bl1]),
        "blue_2": mean([tensor.item() for tensor in bl2]),
        "blue_3": mean([tensor.item() for tensor in bl3]),
        "bert_score": np.mean(np.mean([tensor for tensor in brt])),
        "meteor_score": np.mean([tensor for tensor in mtr]),
        "rouge_score": np.mean([tensor for tensor in rg])
    })

def fit_model(args=Config):
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    device = args.device

    model = ClipCaptionModel(args.prefix_length)
    model = model.to(args.device)

    wandb.watch(model, log_freq=10, log="gradients")

    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate,
                          relative_step=False  # for adafactor
                          )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=20, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=20, shuffle=True, drop_last=False)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=15000
    )
    print("ZERO SHOT")
    evaluate(model, optimizer, scheduler, loss_func, val_loader, args)
    print("Start train model")
    for epoch in range(args.num_epochs):
        if epoch == args.frozen_gpt:
            print("GPT UNFROZEN")
            for p in model.gpt.parameters():
                p.requires_grad = True
        if epoch == args.frozen_clip:
            print("CLIP UNFROZEN")
            for p in model.clip_model.parameters():
                p.requires_grad = True
        print(f"---------- Train epoch {epoch} ---------")
        train(model, optimizer, scheduler, loss_func, train_loader, epoch, args)
        print(f"---------- Evaluate epoch {epoch} ---------")
        evaluate(model, optimizer, scheduler, loss_func, val_loader, args)

if __name__ == "__main__":
    train_dataset = CocoDataset(coef_size=0.5)
    val_dataset = CocoDataset(image_path="data/coco_dataset/val2014",
                              ann_path="data/coco_dataset/annotations/captions_val2014.json",
                              caption_path="data/coco_dataset/coco_val_translation.jsonl", data_type='val', coef_size=0.05)

    with open("wandb_config.json", "r") as file:
        wandb_credetionals = json.loads(file.read())
    wandb.login(relogin=True, key=wandb_credetionals["key"])
    wandb.init(project=wandb_credetionals["project"], sync_tensorboard=True, name=wandb_credetionals["name"])

    fit_model()