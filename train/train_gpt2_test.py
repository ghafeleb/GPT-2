import sys, os
sys.path.insert(0, '../')
print(os.getcwd())
import argparse
import wandb
from model.gpt2 import *
from options import parse_gpt2_train_args, parse_gpt2_eval_args
import tiktoken 
import torch
from optimizer.optimizer_entry import select_optimizer
from data.data_entry import get_dataset_by_type
from data.dataloader import *
from train_gpt2 import *
import time

def get_data_batch(args, device):
    enc = tiktoken.get_encoding('gpt2')
    dataset = get_dataset_by_type(args)
    if args.data_type == 'super_tiny_shakespear':
        dataset = dataset[:1000]
        tokens = enc.encode(dataset)
        B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1])
    buf = buf.to(device)
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    return x, y

def get_logits(device, x, y):
    model = GPT(GPTConfig())
    model.to(device)
    logits = model(x, y)
    print(logits.shape)

def get_logits_and_loss(device, x, y):
    model = GPT(GPTConfig())
    model.to(device)
    logits, loss = model(x, y)
    print(logits.shape)
    print(loss)


def train_simple(args, model, x, y):
    optimizer_f = select_optimizer(args)
    optimizer = optimizer_f(model.parameters(), lr = args.lr)

    for epoch in range(args.epochs):
        t_start = time.time()
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t_end = time.time()
        run_time = (t_end - t_start) * 1000 # Milisecond
        print(f"Epoch {epoch+1}, loss: {loss.item()}, Run time: {run_time:.2f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser = parse_gpt2_train_args(parser)
    parser = parse_gpt2_eval_args(parser)
    args = parser.parse_args()
    if not args.use_wandb:
        wandb.init(project=args.project_name, config=args, mode="disabled")
    else:
        wandb.init(project=args.project_name, config=args)
    device = "cpu"
    if torch.cuda.is_available() and args.device == "cuda":
        device = "cuda" 
    print(f"Running on {device}")
    model = get_model(args, device)
    x, y = get_data_batch(args, device)
    get_logits_and_loss(device, x, y)
    train_simple(args, model, x, y)

if __name__ == '__main__':
    main()