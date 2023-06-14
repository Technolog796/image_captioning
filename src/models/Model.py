import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.amp import autocast
from torch import einsum

from einops import rearrange

import open_clip

from transformers import GPT2LMHeadModel, AutoTokenizer

from typing import Optional

import math

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = nnf.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class BidirectionalCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            context_dim=None,
            dropout=0.,
            talking_heads=False,
            prenorm=False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
            self,
            x,
            context,
            mask=None,
            context_mask=None,
            return_attn=False,
            rel_pos_bias=None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device
        x = self.norm(x)
        context = self.context_norm(context)
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                                           (qk, context_qk, v, context_v))
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device=device, dtype=torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device=device, dtype=torch.bool))
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        attn = stable_softmax(sim, dim=-1)
        context_attn = stable_softmax(sim, dim=-2)
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        context_out = self.context_to_out(context_out)
        if return_attn:
            return out, context_out, attn, context_attn
        return out, context_out


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o


class QFormerBlock(nn.Module):
    def __init__(self, img_emb_size, text_emb_size, output_size, bias=True, act=nn.Tanh):
        super(QFormerBlock, self).__init__()

        self.attn = MultiheadAttention(text_emb_size, text_emb_size, 16)
        self.cross_attn = BidirectionalCrossAttention(
            dim=img_emb_size,
            heads=16,
            dim_head=1024,
            context_dim=text_emb_size
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_emb_size, text_emb_size * 2),
            act(),
            nn.Linear(text_emb_size * 2, text_emb_size * 2),
            act(),
            nn.Linear(text_emb_size * 2, output_size)
        )

    @autocast("cuda")
    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        text_emb = self.attn(text_emb)
        img_emb, text_emb = self.cross_attn(img_emb.reshape(-1, 1, img_emb.shape[1]), text_emb)
        text_emb = self.text_mlp(text_emb)
        return img_emb, text_emb


class MySequential(nn.Sequential):
    def forward(self, *inp):
        for module in self._modules.values():
            inp = module(*inp)
            if inp[0].shape[1] == 1:
                inp = (inp[0][:, 0, :], inp[1])
        return inp


class QFormer(nn.Module):
    def __init__(self, img_emb_size, text_emb_size, output_size, n_blocks=4, bias=True, act=nn.Tanh):
        super(QFormer, self).__init__()

        self.blocks = MySequential(
            *[QFormerBlock(img_emb_size, text_emb_size, text_emb_size) for _ in range(n_blocks)],
        )
        self.res = nn.Sequential(
            nn.Linear(img_emb_size + text_emb_size, output_size)
        )

    @autocast("cuda")
    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        img_emb, text_emb = self.blocks(img_emb, text_emb)
        text_emb = text_emb.mean(axis=1)
        res_emb = torch.cat((img_emb, text_emb), axis=1)
        res_emb = self.res(res_emb)
        return res_emb


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, act=nn.Tanh):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_shape, input_shape * 2),
            act(),
            nn.Linear(input_shape * 2, output_shape)
        )

    @autocast("cuda")
    def forward(self, x):
        return self.seq(x)


def freeze(
        model,
        freeze_emb=False,
        freeze_ln=False,
        freeze_attn=False,
        freeze_ff=False,
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
    def __init__(self, config, prefix_length: int, prefix_size: int = 640, dist_loss=nn.MSELoss()):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.clip_model, _, _ = open_clip.create_model_and_transforms(config.encoder, pretrained="laion400m_e32")
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder)
        self.gpt = GPT2LMHeadModel.from_pretrained(config.decoder,
                                                   eos_token_id=self.tokenizer.pad_token_id)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = QFormer(prefix_size, self.gpt_embedding_size,
                                    self.gpt_embedding_size * prefix_length)
        self.device = config.device
        self.dist_loss = dist_loss
        self.mlp = MLP(self.gpt_embedding_size, self.gpt_embedding_size)

        for p in self.gpt.parameters():
            p.requires_grad = False
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    @autocast("cuda")
    def forward(self, query_tokens: torch.Tensor, query_mask: Optional[torch.Tensor],
                answer_tokens: torch.Tensor, answer_mask: Optional[torch.Tensor], image):
        embedding_text = self.gpt.transformer.wte(query_tokens)
        image = self.clip_model.encode_image(image)
        prefix_projections = self.clip_project(image.float(), embedding_text).view(-1, self.prefix_length,
                                                                                   self.gpt_embedding_size)
        prefix_projections = self.mlp(prefix_projections)
        out = self.gpt(inputs_embeds=prefix_projections, labels=answer_tokens)
        return out, prefix_projections

    def generate(self, image, texts):
        tokens = torch.tensor(self.tokenizer.batch_encode_plus(texts)['input_ids'], dtype=torch.int64).to(self.device)

        embedding_text = self.gpt.transformer.wte(tokens)
        image = self.clip_model.encode_image(image)
        prefix_projections = self.clip_project(image.float(), embedding_text).view(-1, self.prefix_length,
                                                                                   self.gpt_embedding_size)
        prefix_projections = self.mlp(prefix_projections)
        out = self.gpt.generate(
            inputs_embeds=prefix_projections,
            max_length=self.prefix_length,
            no_repeat_ngram_size=3,
            repetition_penalty=2.,
        )
        res = [self.tokenizer.decode(x) for x in out]
        return res
