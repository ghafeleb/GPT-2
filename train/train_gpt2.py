# import sys, os
# print(os.getcwd())
# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)

import sys, os
sys.path.insert(0, '../')
print(os.getcwd())
import argparse
import wandb
from model.gpt2 import *
from options import parse_gpt2_train_args, parse_gpt2_eval_args
import tiktoken 
import torch

def experiment(args):
    device = "cpu"
    if torch.cuda.is_available() and args.device == "cuda":
        device = "cuda" 
    print(f"model_type: {args.model_type}")
    if args.hf_weight:
        model = GPT.from_pretrained(args.model_type)
    else:
        model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    print("Loaded model!")
    return model

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
            logits = model(x) # Get probability of vocabularies for all tokens
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

def get_data_batch():
    enc = tiktoken.get_encoding('gpt2')
    with open('../data/input.txt', 'r') as file:
        text_data = file.read()
    text_data = text_data[:1000]
    tokens = enc.encode(text_data)
    B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1])
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    return x, y

def get_logits(device, x, y):
    model = GPT(GPTConfig())
    model.to(device)
    logits, loss = model(x, y)
    print(logits.shape)
    print(loss)


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
    # if torch.cuda.is_available() and args.device == "cuda":
    #     device = "cuda" 
    print(f"Running on {device}")
    # model = experiment(args)
    # if args.generate_next_tokens:
    #     eval_model(args, model)
    x, y = get_data_batch()
    get_logits(device, x, y)

if __name__ == '__main__':
    main()