import os
import sys
import argparse
import dataclasses
import gc
import glob
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

###
# NanoGPT for MacBook with Metal Backend
# Modified from the original NanoGPT speedrun training script
###

# -----------------------------------------------------------------------------
# Utility functions

def report_mem_consumption():
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
        mem_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 * 1024 * 1024)
        return f'vram:{mem_used:.1f}/{mem_total:.1f}GB'
    else:
        # For Metal backend, we don't have direct memory reporting
        return "metal memory usage unavailable"

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def norm(x):
    return F.layer_norm(x, (x.size(-1),))

def softcap(x, cap=1):
    return cap * F.tanh(x / cap)

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        # Transpose to [B, n_head, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Regular attention - replacement for flex_attention
        # [B, n_head, T, head_dim] @ [B, n_head, head_dim, T] = [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v  # [B, n_head, T, T] @ [B, n_head, T, head_dim] = [B, n_head, T, head_dim]
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # ReLU^2 activation
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / (2 * config.n_layer) ** 0.5

    def forward(self, x):
        x = x + self.attn_scale * self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets):
        B, T = idx.size()
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = norm(x)

        # compute loss
        logits = softcap(self.lm_head(x), cap=30)  # tanh logit softcap
        logits = logits.float()  # use float for logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

# -----------------------------------------------------------------------------
# Simple Data Loader

def _load_data_shard(filename):
    with open(filename, 'rb') as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
        assert header[1] == 1, 'unsupported version'
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, 'number of tokens read does not match header?'
    return tokens

class SimpleDataLoader:
    def __init__(self, filename_pattern, B, T):
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f'did not find any files that match the pattern {filename_pattern}'

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, device):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.advance()
        return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# int main

def main(n_head=12):
    @dataclass
    class Hyperparameters:
        input_bin: str = 'data/tiny_shakespeare_train_*.bin'  # Change to your dataset path
        input_val_bin: str = 'data/tiny_shakespeare_val_*.bin'  # Change to your dataset path
        batch_size = 8  # batch size for training
        sequence_length: int = 256  # reduced from 32k for better Mac compatibility
        num_iterations: int = 1000  # reduced from 3584
        learning_rate: float = 0.0003
        weight_decay: float = 0.01
        val_loss_every: int = 50
        val_tokens: int = 100_000  # reduced from original

    @dataclass
    class GPTConfig:
        vocab_size: int = 50304
        n_layer: int = 6  # reduced from 12 for Mac
        n_head: int = 12  # default, will be overridden
        n_embd: int = 384  # reduced from 768 for Mac

        def __post_init__(self):
            assert self.n_embd % self.n_head == 0
            self.head_dim = self.n_embd // self.n_head

    args = Hyperparameters()
    
    # Use MPS (Metal Performance Shaders) if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal) device: {device}")
    else:
        device = torch.device("cpu")
        print(f"MPS not available, using CPU: {device}")

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    train_accumulation_steps = 1  # simplified - no accumulation for Mac
    val_steps = args.val_tokens // (B * T)
    tokens_per_fwdbwd = B * T

    # begin logging
    os.makedirs('logs', exist_ok=True)
    run_id = uuid.uuid4()
    logfile = f'logs/{run_id}_{n_head}heads.txt'
    print(f"Logging to {logfile}")
    
    def log(s):
        print(s)
        with open(logfile, 'a') as f:
            print(s, file=f)

    log(f'Hyperparameters: {args}')
    log(f'Model Config: n_head={n_head}, n_layer={GPTConfig.n_layer}, n_embd={GPTConfig.n_embd}')

    # init the model from scratch
    model_config = GPTConfig()
    # Override the number of heads
    model_config.n_head = n_head
    # Ensure n_embd is divisible by n_head
    if model_config.n_embd % n_head != 0:
        model_config.n_embd = n_head * (model_config.n_embd // n_head + 1)
        log(f"Adjusted n_embd to {model_config.n_embd} to be divisible by {n_head}")
    model = GPT(model_config).to(device)
    # For Mac, we skip torch.compile
    
    log(f'Model initialized on device: {device}')

    # load tokens
    train_loader = SimpleDataLoader(args.input_bin, B, T)
    val_loader = SimpleDataLoader(args.input_val_bin, B, T)
    
    # init the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # linear learning rate scheduler
    def get_lr(it):
        return max(0.1, 1.0 - (it / args.num_iterations))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # start the clock
    t0 = time.perf_counter()
    tokens_seen = 0
    training_time = 0
    
    # Record losses for plotting
    train_losses = []
    val_losses = []
    timestamps = []
    
    # begin training
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations
        
        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            training_time += time.perf_counter() - t0
            
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch(device)
                    loss = model(x_val, y_val)
                    val_loss += loss.item()
            val_loss /= val_steps
            val_losses.append(val_loss)
            timestamps.append(training_time)
            
            # log val loss
            log(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time:.2f}s')
            
            # start the clock again
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        x, y = train_loader.next_batch(device)
        
        # forward pass
        loss = model(x, y)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        # --------------- TRAINING SECTION END -------------------
        
        # logging
        tokens_seen += tokens_per_fwdbwd
        if step % 10 == 0:
            log(f'step:{step}/{args.num_iterations} train_loss:{loss.item():.4f}')

    # Save results for later comparison
    results = {
        'n_head': n_head,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'timestamps': timestamps,
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'training_time': training_time,
    }
    
    import pickle
    with open(f'results_{n_head}heads.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    log(f'Training complete. Final validation loss: {val_losses[-1] if val_losses else "N/A"}')
    log(f'Total training time: {training_time:.2f}s')
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a small GPT model.')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    args = parser.parse_args()
    
    main(n_head=args.n_head)
