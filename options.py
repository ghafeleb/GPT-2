import argparse
import os
from model.model_entry import model_choice
from random import randint
import datetime
from optimizer.optimizer_entry import optimizer_choice
from data.data_entry import data_choice



def parse_common_args(parser):
    #######################################################
    #   Important Common Arguments
    #######################################################
    parser.add_argument('--model_type', type=str, default='gpt2', choices=model_choice,
                        help='used in model_entry.py')
    parser.add_argument('--data_type', type=str, default='super_tiny_shakespear', choices=data_choice,
                        help='used in data_entry.py')
    parser.add_argument('--device', type=str, default='cuda',
                        help='choose device type: cpu, cuda, mps')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--data_n_worker', type=int, default=0,
    #                     help='number of worker for dataloader')

    #######################################################
    #  Default Arguments, Change as needed
    #######################################################
    parser.add_argument('--project_name', type=str,
                        default='gpt2-pretrain', help='project name')
    # parser.add_argument('--data_dir', type=str, default='./raw_data',
    #                     help='location where data is stored')
    # parser.add_argument('--save_prefix', type=str, default='pref',
    #                     help='some comment for model or test result dir')
    parser.add_argument('--use_wandb', default=False, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--drop_last', default=True,
    #                     action=argparse.BooleanOptionalAction)
    # # checkpoint
    # check_point_dir = os.path.join("./checkpoint", datetime.date.today().strftime('%Y-%m-%d'),
    #                                datetime.datetime.now().strftime('%H_%M_%S'))
    # if not os.path.exists(check_point_dir):
    #     os.makedirs(check_point_dir, exist_ok=True)
    # parser.add_argument('--checkpoint', type=str, default=check_point_dir,
    #                     help='where to check point or model state dict')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)

    # #######################################################
    # #   Pretrain model arguments
    # #######################################################
    # parser.add_argument('--pretrain_epochs', type=int, default=50,
    #                     help='the epochs for performing whole algorithm')
    # parser.add_argument('--pretrain_lr', type=float,
    #                     default=3e-4, help='learning rate')
    # parser.add_argument('--pretrain_optimizer', type=str, default='adam', choices=optimizer_choice,
    #                     help='used in optimizer_entry.py')
    # parser.add_argument('--model_name', type=str,
    #                     default='pretrain_model.pt', help='pretrain model name')

    # #######################################################
    # #   Train model arguments
    # #######################################################
    # # arguments trigger warm start process
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--pretrain_model', type=str,
    #                     default=None, help='path of pre_trained_model')
    # parser.add_argument('--warm_start', action=argparse.BooleanOptionalAction,
    #                     help='if user want to warm-start the algorithm')
    # # common training args
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size used in training")
    parser.add_argument('--token_size', type=int, default=32,
                        help="token size used in training")
    # parser.add_argument('--pub_batch_size', type=int,
    #                     default=1024, help="public batch size used in training")
    # parser.add_argument('--public_private_ratio', type=float,
    #                     default=0.04, help='ratio of public and private data')
    # parser.add_argument('--train_valid_ratio', type=float,
    #                     default=0.2, help="ratio of train and validation dataset")
    # parser.add_argument('--max_grad_norm', type=float, default=1,
    #                     help='max grad norm C for differential privacy')
    # parser.add_argument('--epsilon', type=float, default=25.8,
    #                     help='target epsilon for differential privacy')
    # parser.add_argument('--delta', type=float, default=1e-6,
    #                     help='target delta for differential privacy')
    parser.add_argument('--epochs', type=int, default=50,
                        help='the epochs for performing training')
    # parser.add_argument('--iterations', type=int, default=0,
    #                     help='the iterations for performing training')

    # #######################################################
    # #   Optimizer arguments
    # #######################################################

    # parser.add_argument('--priv_unit_g_p', type=float, default=-1,
    #                     help='parameter p used in priv_unit_G')
    
    # parser.add_argument('--pub_grad_scaler', type=float, default=-1,
    #                     help='public gradient scaler')
    
    parser.add_argument('--optimizer', type=str, default='adam', choices=optimizer_choice,
                        help='used in optimizer_entry.py')

    # parser.add_argument('--semi_dp_beta', type=float, default=0.3,
    #                     help='special hyperparmeter beta for semi dp optimizer')
    # parser.add_argument('--semi_dp_public_norm', type=float, default=None,
    #                     help='special hyperparameter for semi_dp optimizer')

    # parser.add_argument('--pda_pdma_k', type=float, default=500,
    #                     help='special hyperparameter K for pda-pdma optimizer')
    # parser.add_argument('--pda_pdma_alpha', type=float,
    #                     help='special hyperparameter alpha_t for pda-pdma optimizer')

    # parser.add_argument('--momentum', default=0, type=float, metavar='M',
    #                     help='momentum for sgd, alpha parameter for adam')
    # parser.add_argument('--weight_decay', default=0,
    #                     type=float, help='weight decay (default: 0)')

    # #######################################################
    # #   LR and LR scheduler arguments
    # #######################################################
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    # parser.add_argument('--lr_scheduler', type=str, default='explr', choices=[
    #                     'explr', 'steplr', 'cosine', 'plateau', 'cycle'], help="learning rate scheduler")
    # parser.add_argument('--lr_scheduler_gamma', type=float, default=1,
    #                     help="learning rate scheduler hyperparameters gamma (default: 0)")
    # parser.add_argument('--lr_scheduler_step_size', type=int, default=10,
    #                     help="learning rate scheduler hyperparameters step_size (default: 10)")
    # parser.add_argument('--lr_scheduler_T_max', type=int, default=10,
    #                     help="learning rate scheduler hyperparameters T_max (default: 10)")
    # parser.add_argument('--lr_scheduler_eta_min', type=float, default=0,
    #                     help="learning rate scheduler hyperparameters eta_min (default: 0)")
    # parser.add_argument('--lr_scheduler_factor', type=float, default=0.1,
    #                     help="learning rate scheduler hyperparameters factor (default: 0.1)")
    # parser.add_argument('--lr_scheduler_patience', type=int, default=10,
    #                     help="learning rate scheduler hyperparameters patience (default: 10)")
    # parser.add_argument('--lr_scheduler_base_lr', type=float, default=0.00001,
    #                     help="learning rate scheduler hyperparameters base_lr (default: 0.00001)")
    # parser.add_argument('--lr_scheduler_max_lr', type=float, default=0.1,
    #                     help="learning rate scheduler hyperparameters max_lr (default: 0.1)")
    # parser.add_argument('--lr_scheduler_step_size_up', type=int, default=20,
    #                     help="learning rate scheduler hyperparameters step_size_up (default: 20)")
    # parser.add_argument('--lr_scheduler_mode', type=str, default='triangular2',
    #                     help="learning rate scheduler hyperparameters mode (default: triangular2)")
    return parser




def parse_gpt2_train_args(parser):
    parser = parse_train_args(parser)
    #######################################################
    #   Important Common Arguments
    #######################################################
    # parser.add_argument('--linear_reg_p', type=int, default=500,
    #                     help='linear regression p, public samples = p')
    # parser.add_argument('--linear_reg_ratio', type=float, default=1.5,
    #                     help='private data ratio, private_data_size = linear_reg_ratio * linear_reg_p')
    # parser.add_argument('--semi_dp_kappa', type=float, default=0.01, help='kappa for semi dp')
    # parser.add_argument('--pda_pdmd_constant', type=float, default=0.01)

    # # code legacy reason
    # parser.add_argument('--private_epochs', type=int, default=40,
    #                     help='same as epochs')

    return parser


def parse_gpt2_eval_args(parser):
    #######################################################
    #   Important Common Arguments
    #######################################################
    parser.add_argument('--hf_weight', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--generate_next_tokens', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_return_sequences', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=30)
    parser.add_argument('--prefix', type=str, default="Hello, I'm a language model,")
    # parser.add_argument('--linear_reg_ratio', type=float, default=1.5,
    #                     help='private data ratio, private_data_size = linear_reg_ratio * linear_reg_p')
    # parser.add_argument('--semi_dp_kappa', type=float, default=0.01, help='kappa for semi dp')
    # parser.add_argument('--pda_pdmd_constant', type=float, default=0.01)

    # # code legacy reason
    # parser.add_argument('--private_epochs', type=int, default=40,
    #                     help='same as epochs')

    return parser