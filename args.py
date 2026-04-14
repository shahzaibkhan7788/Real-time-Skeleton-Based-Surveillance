import os
import pickle
import time
import argparse
import yaml


def init_args():
    parser = init_parser()
    args = parser.parse_args()
    cfg = load_config(getattr(args, "config", None))
    args = merge_config(args, cfg, parser)
    return init_sub_args(args)

def init_sub_args(args):
    dataset = args.dataset
    
    # 1. Handle Video Paths independently (allow providing only one)
    args.vid_path = {
        'train': args.vid_path_train if args.vid_path_train else os.path.join(args.data_dir, dataset, 'train/images/'),
        'test':  args.vid_path_test  if args.vid_path_test  else os.path.join(args.data_dir, dataset, 'test/frames/')
    }

    # 2. Handle Pose Paths independently (allow providing only one)
    args.pose_path = {
        'train': args.pose_path_train if args.pose_path_train else os.path.join(args.data_dir, dataset, 'pose', 'train/'),
        'test':  args.pose_path_test  if args.pose_path_test  else os.path.join(args.data_dir, dataset, 'pose', 'test/')
    }
    args.pose_path["train_abnormal"] = args.pose_path_train_abnormal
    args.ckpt_dir = None

    # Branch-specific default save directory
    if not args.model_save_dir:
        branch_tag = args.branch.lower().replace("_", "-")
        args.model_save_dir = f"./results-{branch_tag}"
    elif args.model_save_dir in ['./results', 'results', 'results/']:
        branch_tag = args.branch.lower().replace("_", "-")
        args.model_save_dir = os.path.join(args.model_save_dir, f"results-{branch_tag}")

    # Default save_results_dir per branch
    if args.save_results and (not args.save_results_dir):
        branch_tag = args.branch.lower().replace("_", "-")
        args.save_results_dir = f"./evaluation_results_{branch_tag}"

    model_args = args_rm_prefix(args, 'model_')
    return args, model_args


def init_parser(default_data_dir='data/', default_exp_dir='data/exp_dir'):
    parser = argparse.ArgumentParser(prog="STG-NF")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to YAML config file (default: config/default.yaml)')
    
    # General Args
    parser.add_argument('--vid_path_train', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train_abnormal', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train', type=str, default=None, help='Path to training pose')
    parser.add_argument('--vid_path_test', type=str, default=None, help='Path to test vids')
    parser.add_argument('--pose_path_test', type=str, default=None, help='Path to test pose')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech',
                        choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal', 'c1', 'c2', 'c3', 'c4', 'CHAD', 'CPCC0','CPCC1','CPCC2','CPCC3','CPCC4','CPCC5','CPCC6', 'CPCC', 'corridor', 'Avenue', 'NWPUC'], help='Dataset for Eval')
    parser.add_argument('--vid_res', type=str, default=None, help='Video Res')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV', help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999, help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1], help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR', help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=8, metavar='W', help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--plot_vid', type=int, default=0, help='Plot test videos')
    parser.add_argument('--only_test', action='store_true', help='Visualize train/test data')
    parser.add_argument('--mask_root', type=str, help='Path to the gt masks.')

    # Data Params
    parser.add_argument('--num_transform', type=int, default=2, metavar='T', help='number of transformations to use for augmentation (default: 2)')
    parser.add_argument('--num_kp', type=int, default=18, metavar='T', help='number of keypoints in the skeleton structure (default: 17)')
    parser.add_argument('--headless', action='store_true', help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--norm_scale', '-ns', type=int, default=0, metavar='NS', choices=[0, 1], help='Scale without keeping proportions [1/0] (default: 0)')
    parser.add_argument('--prop_norm_scale', '-pns', type=int, default=1, metavar='PNS', choices=[0, 1], help='Scale keeping proportions [1/0] (default: 1)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH', help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=24, metavar='SGLEN', help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=12, metavar='SGST', help='Stride for training segment sliding window')
    parser.add_argument('--specific_clip', type=int, default=None, help='Train and Eval on Specific Clip')
    parser.add_argument('--global_pose_segs', action='store_false', help='Use unormalized pose segs')
    parser.add_argument('--relative', action='store_true', help='Adding relative movement to the data')
    parser.add_argument('--traj', action='store_true', help='Adding global trajectory')
    
    # Model Params
    parser.add_argument('--model_ckpt_dir', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--model_ckpt_F', type=str, metavar='model', help="Path to a pretrained SPARTA_F")
    parser.add_argument('--model_ckpt_C', type=str, metavar='model', help="Path to a pretrained SPARTA_C")
    parser.add_argument('--model_save_dir', type=str, default=None, metavar='model', help="Directory to save checkpoints and logs")
    parser.add_argument('--batch_size', type=int, default=256,  metavar='B', help='Batch size for train')
    parser.add_argument('--epochs', '-model_e', type=int, default=50, metavar='E', help = 'Number of epochs per cycle')
    parser.add_argument('--model_optimizer', '-model_o', type=str, default='adam', metavar='model_OPT', help = "Optimizer")
    parser.add_argument('--model_lr', type=float, default=5e-10, metavar='LR', help='Optimizer Learning Rate Parameter')
    parser.add_argument('--model_weight_decay', '-model_wd', type=float, default=5e-5, metavar='WD', help='Optimizer Weight Decay Parameter')
    parser.add_argument('--model_lr_decay', '-model_ld', type=float, default=0.99, metavar='LD', help='Optimizer Learning Rate Decay Parameter')
    parser.add_argument('--model_latent_dim', default=64, type=int, help='size of latent dimension')
    parser.add_argument('--model_num_layers', default=4, type=int, help='number of hidden layers')
    parser.add_argument('--model_num_heads', default=2, type=int, help='Number of heads in the transformer module')
    parser.add_argument('--sched', type=str, default='tri', metavar='AE_SCH', help="Optimization LR scheduler (default: 'tri')")
    parser.add_argument('--model_loss', type=str, default='mse', choices=['mse', 'emd', 'spec', 'ssim', 'cos', 'CHAD'], help='Loss function')
    parser.add_argument('--model_a', type=float, default=1.0, metavar='a', help='MSE multiplier')
    parser.add_argument('--model_b', type=float, default=1.0, metavar='b', help='Cosine multiplier')
    parser.add_argument('--model_c', type=float, default=1.0, metavar='c', help='EMD multiplier')
    parser.add_argument('--model_d', type=float, default=1.0, metavar='d', help='Spectural Distance multiplier')
    parser.add_argument('--recon_encoder_path', type=str, help="Path loading the reconstruction encoder")
    parser.add_argument('--save_results', action='store_true', help='save the scores')
    parser.add_argument('--save_results_dir', type=str, help="Path for saving scores")
    parser.add_argument('--token_config', type=str, default='t', choices=['t', 'pst', 'kps', '2ds', 'st'], help='Token Configurations')
    parser.add_argument('--branch', type=str, default='SPARTA_C', choices=['SPARTA_C', 'SPARTA_F', 'SPARTA_H'], help='Choose a specific branch.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Workflow mode: train or test.')
    parser.add_argument('--no_metrics', action='store_true', help='Skip metric computation (no masks).')
    parser.add_argument('--eer_threshold_c', type=float, default=None, help='EER threshold for SPARTA_C')
    parser.add_argument('--eer_threshold_f', type=float, default=None, help='EER threshold for SPARTA_F')
    
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DROPOUT', help='Dropout training Parameter (default: 0.3)')

    # Training workflow controls
    parser.add_argument('--disable_eval_during_train', action='store_true', help='Skip mid-epoch validation; final epoch is always evaluated')
    parser.add_argument('--eval_interval', type=int, default=0, help='Evaluate every N epochs; 0 means only final epoch')
    parser.add_argument('--skip_final_eval', action='store_true', help='--relativeSkip the final evaluation step after training')
    parser.add_argument('--save_all_checkpoints', action='store_true', help='Save a checkpoint file at the end of every epoch')
    parser.add_argument('--metrics_csv', type=str, default='metrics.csv', help='Filename (within model_save_dir) for per-epoch metrics log')

    # default: do evaluate during training unless user opts out
    parser.set_defaults(disable_eval_during_train=False)
    return parser


def load_config(path):
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Config file '{path}' not found; using CLI/default values.")
        return {}
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        print(f"Config file '{path}' is not a mapping; ignoring.")
        return {}
    return data


def merge_config(args, cfg, parser):
    """
    Fill argparse Namespace with values from cfg only when the user
    did not override them on the CLI (detected via parser defaults).
    """
    if not cfg:
        return args
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, 'dest')}
    for key, val in cfg.items():
        if val is None:
            continue
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        default = defaults.get(key, None)
        # If the current value equals the parser default, assume not overridden.
        if current == default:
            setattr(args, key, val)
    return args


def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            model_key = key[len(prefix):]
            wp_args_dict[model_key] = value

    return wp_args


def create_exp_dirs(experiment_dir, dirmap=''):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    dirs = [experiment_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return experiment_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)


def save_dataset(dataset, fname):
    with open(fname, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(fname):
    with open(fname, 'rb') as file:
        return pickle.load(file)
