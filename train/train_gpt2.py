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

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_model(args, device):
    print(f"model_type: {args.model_type}")
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

    for epoch in range(args.epochs):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, loss: {loss.item()}")

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