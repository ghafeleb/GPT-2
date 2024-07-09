# import sys, os
# sys.path.insert(0, '../')
from dataclasses import dataclass # Classes that store information that will be passed between different parts of a program or a system
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Multi-Head Attention in One Class
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, print("Embedding dimension should be divisible by number of heads.")
        # Batch of all key, query, and value projections for all the heads
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3) # *3 because of k, q, v
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B: batch size
        # T: sequence length
        # C: embedding dimension (n_embd)
        B, T, C = x.size()
        # Get q, k, v
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # head size = C // n_head = n_embd // n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # Shape: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # Shape: (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # Shape: (B, nh, T, hs)
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v # Aggregate all the attentions
        # Concatenate all heads
        # contiguous changes the layout of elements in the memory. More info: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc  = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # Feed-Forward Network

    def forward(self, x):
        # Residual is separate from the layer norm >> Gradients flows through blocks and residuals separately
        # Tokens are being handled separately
        x = x + self.attn(self.ln_1(x)) # Attention works as a reduce
        x = x + self.mlp(self.ln_2(x)) # MLP works as the map
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024      # the maximum lenght of the sequence (# of tokens)
    vocab_size: int = 50257     # number of tokens in the vocabulary: 50,000 BPE merges, 256 butes tokens, 1 eos token
    n_layer: int = 12           # Number of layers
    n_head: int = 12            # Number of heads
    n_embd: int = 768           # Dimension of embedding

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Holds submodules in a dictionary.
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # nn.Embedding: wrapper module around the tensor
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for i in range(config.n_layer)]), # ModuleList allows indexing layers in it
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size
        # Position embedding, shape: (T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape: T
        pos_emb = self.transformer.wpe(pos)
        # Token embedding, shape: (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # Final layer norm
        x = self.transformer.ln_f(x)
        # Classifier
        logits = self.lm_head(x) # Shape: (B, T, vocab_size)
        return logits


    @classmethod
    def from_pretrained(cls, model_type):
        """
            Load the pretrained parameters of GPT2 from huggingface
        """
        from transformers import GPT2LMHeadModel
        print("Loading weights from the pretrained GPT-2 model: ", model_type)

        # model_type defines the n_layer, n_head, and n_embd
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # Size: 124M parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # Size: 350M parameters
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # Size: 774M parameters
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # Size: 1558M parameters
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # Create GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Hugging Face GPT
        model_hugging_face = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hugging_face = model_hugging_face.state_dict()

        # Copy hugging face params and double check the parameters to align        
        sd_keys_hf = sd_hugging_face.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hugging_face[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hugging_face[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hugging_face[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hugging_face[k])

        return model





