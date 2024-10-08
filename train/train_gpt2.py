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
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_model(args, device):
    print(f"model_type: {args.model_type}")
    torch.set_float32_matmul_precision(args.matmul_precision)
    # if args.flash_attention:
    #     model_class = GPTFlashAttention
    # else:
    model_class = GPT
    if not args.train and args.hf_weight:
        model = model_class.from_pretrained(args.model_type)
    else:
        # model = model_class(GPTConfig())
        model = model_class(GPTConfig(flash_attention = args.flash_attention, 
                                      vocab_size = args.vocab_size))

    if args.train:
        model.train()
    else:
        model.eval()

    model.to(device)
    print(f"args.compile_model: {args.compile_model}")
    if args.compile_model:
        model = torch.compile(model)
    print("Loaded model!")
    return model

def get_lr(args, it):
    min_lr = args.lr_scheduler_max_lr * 0.1
    if it < args.lr_scheduler_warmup_steps:
        return args.lr_scheduler_max_lr * (it + 1) / args.lr_scheduler_warmup_steps
    if it > args.lr_scheduler_max_steps:
        return min_lr
    decay_rate = (it - args.lr_scheduler_warmup_steps) / (args.lr_scheduler_max_steps - args.lr_scheduler_warmup_steps)
    assert decay_rate >= 0 and decay_rate <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate))
    return min_lr + coeff * (args.lr_scheduler_max_lr - min_lr)

def get_grad_accum_steps(args, ddp_world_size, master_process):
    if args.total_batch_size != -1:
        assert args.total_batch_size % (args.batch_size * args.token_size * ddp_world_size) == 0
        grad_accum_steps = args.total_batch_size // (args.batch_size * args.token_size * ddp_world_size)
        if master_process:
            print(f"total desired batch size: {args.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    else:
        grad_accum_steps = 1
    return grad_accum_steps
def get_dds_setting(args):
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def train(args, model, ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process):
    # Optimizer
    optimizer_f = select_optimizer(args)
    if args.optimizer == 'adam':
        if args.gpt3_adam_beta:
            optimizer = optimizer_f(model.parameters(), lr = args.lr, betas=(0.9, 0.95), eps=1e-8)
        elif args.gpt3_adam_parameters:
            optimizer = model.configure_optimizers(weight_decay = args.weight_decay, learning_rate = args.lr, device_type = device)
        else:
            optimizer = optimizer_f(model.parameters(), lr = args.lr)
    else:
        optimizer = optimizer_f(model.parameters(), lr = args.lr)

    grad_accum_steps = get_grad_accum_steps(args, ddp_world_size, master_process)

    print("GPU ", ddp_rank)
    sys.exit(0)
    # DataLoader
    train_loader = DataLoaderLite(args)

    autocast_type = torch.bfloat16 if args.autocast_type == 'bf16' else torch.float32        
    for epoch in range(args.epochs):
        t_start = time.time()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=autocast_type):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if args.clip_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.lr_scheduler == 'cosine':
            lr = get_lr(args, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t_end = time.time()
        run_time = (t_end - t_start) * 1000 # Milisecond
        tokens_per_second = (train_loader.B * train_loader.T * grad_accum_steps ) / (t_end - t_start)
        print(f"\nEpoch {epoch+1} | loss: {loss.item():.6f} | Run time: {run_time:.2f} ms | token/sec: {tokens_per_second:.2f}", end = " ")
        if args.clip_grad_norm:
            print(f" | norm: {norm:.4f}", end = " ")
        if args.lr_scheduler:
            print(f" | lr: {lr:.4e}", end = " ")
            
def main():
    parser = argparse.ArgumentParser()
    parser = parse_gpt2_train_args(parser)
    parser = parse_gpt2_eval_args(parser)
    args = parser.parse_args()
    if not args.use_wandb:
        wandb.init(project=args.project_name, config=args, mode="disabled")
    else:
        wandb.init(project=args.project_name, config=args)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = get_dds_setting(args)
    # device = "cpu"
    # if torch.cuda.is_available() and args.device == "cuda":
    #     device = "cuda" 
    print(f"Running on {device}")
    set_seed(args.seed)
    model = get_model(args, device)
    train(args, model, ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process)

if __name__ == '__main__':
    main()
