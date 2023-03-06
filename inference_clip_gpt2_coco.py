import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import cv2
from PIL import Image
from typing import Tuple, Optional, Union

import clip

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    #@autocast()  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    
def freeze(
    model,
    freeze_emb=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_other=False,
):
    
    for name, p in model.named_parameters():
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
           
    return model

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 768):
          super(ClipCaptionModel, self).__init__()
          self.prefix_length = prefix_length
          """
          ru gpts shit
          
          """
          self.gpt = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
          # self.gpt = AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-smal')
          
          self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
          self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                  self.gpt_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    #@autocast() 
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix.float()).view(-1, self.prefix_length, self.gpt_embedding_size)

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def filter_ngrams(output_text):
    a_pos = output_text.find(' Ответ:')
    sec_a_pos = output_text.find(' Ответ:', a_pos + 1)
    return output_text[:sec_a_pos]

def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt='',
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.98,
        temperature=1.,
        stop_token = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            if not tokens:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    
            emb_tokens = model.gpt.transformer.wte(tokens)
            
            if embed is not None:
                generated = torch.cat((embed, emb_tokens), dim=1)
            else:
                generated = emb_tokens

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                top_k = 2000 
                top_p = 0.98
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
               
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            
            output_text = tokenizer.decode(output_list)
            utput_text = filter_ngrams(output_text)
            generated_list.append(output_text)

    return generated_list[0]

def read_image(path):
    image = cv2.imread(path)
    
    size = 196, 196
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    
    return image

def create_emb(path):
    text = "Вопрос: что изображено на этой картинке? Ответ: "
    image = read_image(path)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    return (prefix, text)

def get_caption(prefix, prompt=''):
    prefix = prefix.to(device)
    with torch.no_grad():
        
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if prompt:
            generated_text_prefix = generate2(model, tokenizer, prompt=prompt, embed=prefix_embed)
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix.replace('\n',' ')

def get_ans(clip_emb, prompt, image_url):
    output = get_caption(clip_emb, prompt=prompt)
    ans = output[len(prompt):].strip()
    return {'image_url': image_url, 'answer': ans}

device = 'cuda:0'
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
prefix_length = 50
model_path = 'checkpoints/prefix_small-001.pt'
model = ClipCaptionPrefix(prefix_length)
model.load_state_dict(torch.load(model_path, map_location='cpu')) 
model.to(device)
print("Model loaded.")

input_image_path = "data/coco_dataset/val2017/000000000285.jpg"

prefix, text = create_emb(input_image_path)
print("Embedding generated.")

ans = get_ans(prefix, text, input_image_path)
print(ans)