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
import time

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_model(args, device):
    print(f"model_type: {args.model_type}")
    torch.set_float32_matmul_precision(args.matmul_precision)
    if not args.train and args.hf_weight:
        model = GPT.from_pretrained(args.model_type)
    else:
        model = GPT(GPTConfig())

    if args.train:
        model.train()
    else:
        model.eval()

    model.to(device)
    print("Loaded model!")
    return model

def train(args, model, device):
    # Optimizer
    optimizer_f = select_optimizer(args)
    optimizer = optimizer_f(model.parameters(), lr = args.lr)

    # DataLoader
    train_loader = DataLoaderLite(args)

    torch.set_float32_matmul_precision(args.matmul_precision)
    for epoch in range(args.epochs):
        t_start = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
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
    set_seed(args.seed)
    model = get_model(args, device)
    train(args, model, device)

if __name__ == '__main__':
    main()