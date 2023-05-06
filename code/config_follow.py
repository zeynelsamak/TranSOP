import argparse
import os
import warnings
import math

warnings.simplefilter(action='ignore', category=FutureWarning)



def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset
    parser.add_argument("--dataset", type=str, default="mrclean_register2population",
                        help="mrclean_register2population")
    parser.add_argument("--data", type=str, help="path to dataset", default=os.getcwd())

    parser.add_argument('--follow_time', type=int, default=2, help='use clinic features')

    # DataLoader
    parser.add_argument('--train_csv', type=str, default='update_base_train.csv')
    parser.add_argument('--val_csv', type=str, default='update_base_val.csv')
    parser.add_argument('--test_csv', type=str, default='update_base_test.csv')
    parser.add_argument("--batch_size", default=16, type=int, help="batch size per gpu")
    parser.add_argument('--add_skull', type=int, default=1, help='using skull stripped images')
    parser.add_argument('--num_patches', type=int, default=4, help='number of patch per volume')
    parser.add_argument('--patch_size', type=int, default=16, help='patch_size')
    parser.add_argument('--file', type=str, default='/user/home/zs18923/data_folds/', help='files')
    parser.add_argument('--folds', type=int, default='1', help='fold')

    parser.add_argument('--clinic', type=int, default=0, help='use clinic features')
    parser.add_argument('--recon', type=int, default=0, help='use clinic features')

    parser.add_argument('--augment', type=float, default=0.5,
                        help='augment')

    # Envs
    parser.add_argument('--device', type=int, default=0, help='device no')
    parser.add_argument('--local', action='store_true', help='run on local machine')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')  # 1453
    parser.add_argument('--pin_memory', action='store_true',
                        help='pin memory ')

    # Model
    parser.add_argument('--model', type=str, default='MoCo')
    parser.add_argument('-a', '--arch', type=str, default='disc')
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--attention', type=int, default=0, help='using attention')
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--out_features", type=int, default=512)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument('--method', type=str, default='NTX', help='choose method')

    # VQ-VAE
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=256)

    # classification
    parser.add_argument("--num_classes", type=int, default=2)

    # Distributed Trianing
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:6666', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cosine', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # Trainer & Optimizer
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['lars', 'adamw', 'adam', 'sgd'],
                        help='optimizer used (default: adam)')
    parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
    parser.add_argument("--num_epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                        help='number of warmup epochs')

    # Checkpoints
    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40,
                        help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10,
                        help='save frequency')

    # Restart
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")

    # Logs
    parser.add_argument("--ckpt_path", type=str, help="path to ckpt")
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
    parser.add_argument('--feature_size', default=48, type=int, help='embedding size')
    parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
    parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')


    opt = parser.parse_args()

    opt.early_stop = 40 // opt.val_freq

    # set the path according to the environment
    # '/home/pgrads/zs18923/linux/zs18923/results/
    if opt.local:
        out_pth = '/run/media/zs18923/zs18923_SSD/results/LastChapter/'
        opt.data_root = '/run/media/zs18923/zs18923_SSD/{}'.format(opt.dataset)
    else:
        out_pth = '/user/work/zs18923/results/FinalChapter/'
        opt.data_root = '/user/work/zs18923/data/{}'.format(opt.dataset)

    opt.root = out_pth+'{0}/'.format(opt.model)

    opt.model_name = '{}_{}_{}_{}_lr_{}_bsz_{}_drop_{}_att_{}_skull_{}_ps{}_np{}_ft{}_trial_{}'.\
        format(opt.arch, opt.loss, opt.dataset, opt.optimizer, opt.learning_rate,
               opt.batch_size, opt.dropout, opt.attention, opt.add_skull, opt.patch_size, opt.num_patches, opt.follow_time, opt.trial)

    opt.default_root_dir = os.path.join(opt.root, opt.model_name)
    if not os.path.isdir(opt.default_root_dir):
        os.makedirs(opt.default_root_dir)

    opt.test_folder = os.path.join(opt.root, opt.model_name, 'test_images')
    if not os.path.isdir(opt.test_folder):
        os.makedirs(opt.test_folder)
    
    opt.val_folder = os.path.join(opt.root, opt.model_name, 'val_images')
    if not os.path.isdir(opt.val_folder):
        os.makedirs(opt.val_folder)

    # opt.save_folder = os.path.join(opt.root, opt.model_name, 'models')
    # opt.ckpt_path = opt.save_folder
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)

    opt.train_csv = opt.file+opt.train_csv
    opt.val_csv = opt.file+opt.val_csv
    opt.test_csv = opt.file+opt.test_csv

    # opt.result_folder = os.path.join(opt.root, opt.model_name, 'results')
    # if not os.path.isdir(opt.result_folder):
    #     os.makedirs(opt.result_folder)
    # opt.img_size = [128,192,128]
    opt.img_size = [32, 192, 128]
    opt.spacing = [3, 1, 1]

    opt.hu_range = [40,80]
    opt.input_shape= [32,192,128]
    opt.resample = [1.0, 1.0, 3.0]


    # warm-up for large-batch training,
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.00001
        opt.warm_epochs = 5
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
            1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt
