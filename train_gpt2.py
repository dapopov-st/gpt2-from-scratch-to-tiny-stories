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

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads a pretrained model from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'Loading weights from  pretrained gpt2: {model_type}...')

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), #124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), #345M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), #774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600) #1558M
        }[model_type]

        config_args['vocab_size'] = 50257 # all gpt2 model checkpoints have 50257 vocab size
        config_args['block_size'] = 1024

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)  
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # don't load lm_head weights

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        # copy over the weights that match in both models
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('.attn.masked_bias')] # just a buffer, ignore
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('attn.bias')] # just a buffer, ignore
        transposed  = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use "Conv1D" module, but we want to use plain vanilla Linear layer
        # so transpose these weights when we import them
        assert len(sd_keys) == len(sd_hf_keys), f"expected {len(sd_keys)} keys, got {len(sd_hf_keys)}"
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"expected {sd_hf[k].shape} to be transposed shape of {sd[k].T.shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd[k].shape == sd_hf[k].shape, f"expected {sd[k].shape} to be equal to {sd_hf[k].shape}"
                sd[k].copy_(sd_hf[k])
        return model

