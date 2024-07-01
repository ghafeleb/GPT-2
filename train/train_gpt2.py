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
from options import parse_gpt2_train_args

def experiment(args):
    print(f"model_type: {args.model_type}")
    model = GPT.from_pretrained(args.model_type)
    print("Worked!")



def main():
    parser = argparse.ArgumentParser()
    parser = parse_gpt2_train_args(parser)
    args = parser.parse_args()
    if not args.use_wandb:
        wandb.init(project=args.project_name, config=args, mode="disabled")
    else:
        wandb.init(project=args.project_name, config=args)
    print(f"Running on {args.device}")
    experiment(args)


if __name__ == '__main__':
    main()