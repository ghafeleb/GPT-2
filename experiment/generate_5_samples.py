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
from train.train_gpt2 import *

def tokenize(args):
    device = "cpu"
    if torch.cuda.is_available() and args.device == "cuda":
        device = "cuda" 
    # Check tiktokenize online here: https://tiktokenizer.vercel.app/
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(args.prefix)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(args.num_return_sequences, 1)
    x = tokens.to(device)
    return x

def generate_next_token(args, model, x):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    while x.size(1) < args.max_length:
        with torch.no_grad():
            logits, _ = model(x) # Get probability of vocabularies for all tokens
            logits = logits[:, -1, :] # Get logits at the last position and discard previous ones, shape: (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # Probabilities of next token
            topk_probs, topk_indecies = torch.topk(probs, 50, dim=-1) # Keep the top 50 probabilies and remove the rest. Then, normalize between what we have.
            ix = torch.multinomial(topk_probs, 1) # Shape: (B, 1)
            xcol = torch.gather(topk_indecies, -1, ix) # (B, 1)
            x = torch.cat((x, xcol), dim=1)
    return x

def decode_tokens(args, x):
    enc = tiktoken.get_encoding('gpt2')
    for i in range(args.num_return_sequences):
        tokens = x[i, :args.max_length].tolist()
        decoded  = enc.decode(tokens)
        print(">> ", decoded)


def eval_model(args, model):
    x = tokenize(args)
    x = generate_next_token(args, model, x)
    decode_tokens(args, x)

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
    model = experiment(args, device)
    if args.generate_next_tokens:
        eval_model(args, model)

if __name__ == '__main__':
    main()