from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # 256 bytes tokens + 1 <|endoftext|> token + 50,000 BPE tokens
    n_layer: int = 12 # number of transformer layers 
    n_head: int = 12 # number of heads in the multiheadattention models
    n_embd: int = 784 # embedding dimension

class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        #key, qury, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask to prevent attention to future tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) # becomes available as self.bias
    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, value for all heads in batch
        qkv = self.c_attn(x) # (B,T, self.n_embd) x (self.n_embd,self.n_embd*3) = (B,T,self.n_embd*3)
        q, k, v  = qkv.split(self.n_embd, dim=2) # (B,T,self.n_embd) x 3; make each split size self.n_embd by splitting dim 2
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        # attention materializes a large (T,T) matrix fo each query and key
        att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1))) # (B, nh, T, hs) x (B, nh, hs, T) = (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
        # Change (B, nh, T, hs) to (B, T, nh, hs) with transpose, reassemle in memory, (B,T,C) makes nh*hs = n_embd (C)
        y = y.transpose(1,2).contiguous().view(B, T, C) 
        # output projection: additional learnable transformation
        y = self.c_proj(y) # (B, T, C)@(C, C) = (B, T, C)
        return y
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x    
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config    
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # With nn.ModuleDict() index into submodules just like a dictionary
        self.tranformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
