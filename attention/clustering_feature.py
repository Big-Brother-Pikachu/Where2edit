import sys
sys.path.append("..")

import argparse
import math
import os
import re
import random
import builtins
import shutil
import datetime
import json
import pickle
import numpy as np

import torch
import torchvision
from torch import optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.cluster import KMeans

from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from attention_model import Generator, StyledConv, PixelNorm, EqualLinear, EqualConv2d
import clip
from utils import ensure_checkpoint_exists


STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def descripition_corpus(args):
    phras = []
    for root, _, files in os.walk('../celeba-caption'):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                choose_line = torch.randint(10, (2, ))
                all_lines = f.readlines()
                for line in [all_lines[id_line] for id_line in choose_line]:
                    line = re.split('[,.]', line.rstrip('\n'))[:-1]
                    line = [phra[5:] if phra.startswith(' and') else phra.lstrip(' ') for phra in line]
                    phras.extend(line)
            # if len(phras) >= 10000:
            #     break

    # phras = []
    # with open("../face2text_v1.0/raw.json",'r') as f:
    #     load_dict = json.load(f)

    # for line in load_dict:
    #     line = re.split('[,.]', line['description'].rstrip('\n'))[:-1]
    #     line = [phra[5:] if phra.startswith(' and') else phra.lstrip(' ') for phra in line]
    #     phras.extend(line)
    
    return phras


class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class Multiply(torch.nn.Module):
    def __init__(self, scale=1):
        super(Multiply, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Addnoise(torch.nn.Module):
    def __init__(self, sigma=1):
        super(Addnoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x  + torch.randn_like(x) * self.sigma
        else:
            return x


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class Gumbel_softmax(torch.nn.Module):
    def __init__(self, temperature=1):
        super(Gumbel_softmax, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        y = gumbel_softmax_sample(x, self.temperature)
        if self.training:
            return y
        else:
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            return y_hard


class Logger(object):
    def __init__(self, stdout, filename):
        self.logfile = filename
        self.terminal = stdout
 
    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, 'a')
                self.log.write(message)
                self.log.close()
            except:
                pass
 
    def flush(self):
        pass


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 


def pairwise_distance(data1, data2=None):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1 

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis


def forgy(X, n_clusters):
	indices = torch.multinomial(torch.ones(X.shape[0], ), n_clusters)
	initial_state = X[indices]
	return initial_state


def lloyd(X, n_clusters, tol=1e-4):
	initial_state = forgy(X, n_clusters)

	while True:
		dis = pairwise_distance(X, initial_state)

		choice_cluster = torch.argmin(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(n_clusters):
			selected = torch.nonzero(choice_cluster==index).squeeze()

			selected = torch.index_select(X, 0, selected)
			initial_state[index] = selected.mean(dim=0)
		

		center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

		if center_shift ** 2 < tol:
			break

	return choice_cluster, initial_state


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # cudnn.benchmark = False

    # set random seed before init model
    # torch.set_deterministic(True)
    cudnn.deterministic = True
    cudnn.benchmark = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = len(args.gpu_id.split(','))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(int(args.gpu_id), ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    set_random_seed(args.seed + args.rank)
    # torch.set_deterministic(True)
    
    if args.rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        dateTime_p = datetime.datetime.now()
        dateTime_p = datetime.datetime.strftime(dateTime_p, '%Y-%m-%d-%H-%M-%S')
        exp_name = args.description.replace(' ', '-') + '-' + dateTime_p
        writer = SummaryWriter(os.path.join(args.results_dir + '/logs', exp_name))
        output_dir = os.path.join(args.results_dir + '/outputs', exp_name)
        os.makedirs(output_dir, exist_ok=True)
        files = ['./clustering_feature.py', './attention_model.py']
        for f in files:
            shutil.copy(f, os.path.join(output_dir, f.split('/')[-1]))
        stdout_backup = sys.stdout
        sys.stdout = Logger(stdout_backup, os.path.join(output_dir, 'run.log'))
        print('--------args----------')
        for k in list(vars(args).keys()):    
            print('%s: %s' % (k, vars(args)[k]))
        print('--------args----------\n')

    ensure_checkpoint_exists(args.ckpt)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    g_ema = Generator(args.stylegan_size, 512, 8, channel_multiplier=args.channel_multiplier)
    w_code_num = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13]
    if args.gpu is None:
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        g_ema.load_state_dict(torch.load(args.ckpt, map_location=loc)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda(args.gpu)
    mean_latent = g_ema.mean_latent(4096)

    batch = args.batch_size
    attention_layer = args.attention_layer
    clusters = args.cluster_num
    if args.latent_path:
        latent_code_init_load = torch.load(args.latent_path).cuda()

    pbar = range(args.step)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    clustering_features = []

    # with open(os.path.join(os.path.join(args.results_dir + '/outputs', 'Beard-2022-04-15-15-35-48'), 'k_means_human_12_layer_10_clusters.pkl'), "rb") as f:
    #     initial_state = pickle.load(f)
    #     # initial_state = torch.from_numpy(kmeans.cluster_centers_).cuda()
    # clusters = 10
    # attention_layer = 13
    # channel_nums = initial_state.shape[-1]
    
    for i in pbar:
        t = i / args.step

        if args.latent_path:
            code_choose = torch.randint(len(latent_code_init_load), (batch, ))
            latent_code_init = latent_code_init_load[code_choose]
        else:
            # latent_code_init = mean_latent.detach().clone().repeat(batch, 18, 1)
            latent_code_init_not_trunc = torch.randn(batch, 512).cuda()
            with torch.no_grad():
                _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                            truncation=args.truncation, truncation_latent=mean_latent)

        with torch.no_grad():
            img_orig, _, _, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True)

        if args.work_in_stylespace:
            with torch.no_grad():
                _, _, latent_code_init, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True)
            latent = [s.detach().clone() for s in latent_code_init]
            # for c, s in enumerate(latent):
            #     if c in STYLESPACE_INDICES_WITHOUT_TORGB:
            #         s.requires_grad = True
        else:
            latent = latent_code_init.detach().clone()

        blend_feature = feature_map[attention_layer - 1]
        size = blend_feature.shape[2] * 2
        position_channel = blend_feature.shape[1] // 16
        x_position = torch.arange(size).cuda().float().unsqueeze(0).repeat(size, 1) * 2 / float(size-1) - 1
        y_position = torch.arange(size).cuda().float().unsqueeze(1).repeat(1, size) * 2 / float(size-1) - 1
        x_position = x_position.unsqueeze(0).unsqueeze(0).repeat(blend_feature.shape[0], position_channel, 1, 1)
        y_position = y_position.unsqueeze(0).unsqueeze(0).repeat(blend_feature.shape[0], position_channel, 1, 1)
        concat_feature = [torch.nn.functional.interpolate(blend_feature.detach(), size=size, mode='bilinear', align_corners=True)]
        # for j in [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]:
        #     blend_feature = torch.nn.functional.interpolate(feature_map[j].detach().clone(), size)
        #     concat_feature.append(blend_feature)
        concat_feature.extend([x_position, y_position])
        concat_feature = torch.cat(concat_feature, dim=1)
        channel_nums = concat_feature.shape[1]
        clustering_features.append(concat_feature.detach().cpu())
        del concat_feature
    total_features = torch.cat(clustering_features, dim=0)
    total_features = total_features.permute(0, 2, 3, 1).contiguous()
    total_features = total_features.view(-1, channel_nums).numpy()

    arr = total_features
    kmeans = KMeans(n_clusters=clusters, random_state=42).fit(arr)
    initial_state = torch.from_numpy(kmeans.cluster_centers_)
    with open(os.path.join(output_dir, 'k_means_human_12_layer_10_clusters.pkl'), "wb") as f:
        pickle.dump(initial_state, f)
    initial_state = initial_state.cuda()

    # choice_cluster, initial_state = lloyd(total_features, clusters)

    # del total_features

    if args.latent_path:
        code_choose = torch.randint(len(latent_code_init_load), (4, ))
        latent_code_init = latent_code_init_load[code_choose]
    else:
        latent_code_init_not_trunc = torch.randn(4, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
        latent = [s.detach().clone() for s in latent_code_init]
    else:
        latent = latent_code_init.detach().clone()

    with torch.no_grad():
        img_orig, _, _, feature_map = g_ema([latent], input_is_latent=True, randomize_noise=False, return_features=True, input_is_stylespace=args.work_in_stylespace)

    colors = (torch.rand(clusters, 3) - 0.5) * 2
    blend_feature = feature_map[attention_layer - 1]
    size = blend_feature.shape[2] * 2
    x_position = torch.arange(size).cuda().float().unsqueeze(0).repeat(size, 1) * 2 / float(size-1) - 1
    y_position = torch.arange(size).cuda().float().unsqueeze(1).repeat(1, size) * 2 / float(size-1) - 1
    x_position = x_position.unsqueeze(0).unsqueeze(0).repeat(blend_feature.shape[0], position_channel, 1, 1)
    y_position = y_position.unsqueeze(0).unsqueeze(0).repeat(blend_feature.shape[0], position_channel, 1, 1)
    concat_feature = [torch.nn.functional.interpolate(blend_feature.detach(), size=size, mode='bilinear', align_corners=True)]
    # for j in [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]:
    #     blend_feature = torch.nn.functional.interpolate(feature_map[j].detach().clone(), size)
    #     concat_feature.append(blend_feature)
    concat_feature.extend([x_position, y_position])
    concat_feature = torch.cat(concat_feature, dim=1)
    concat_feature = concat_feature.permute(0, 2, 3, 1).contiguous()
    concat_feature = concat_feature.view(-1, channel_nums)
    dis = pairwise_distance(concat_feature, initial_state)
    choice_cluster = torch.argmin(dis, dim=1).view(4, size, size)
    blend_choice = torch.ones((4, size, size, 3))
    for i in range(clusters):
        blend_choice[choice_cluster==i] = colors[i]
    blend_choice = torch.nn.functional.interpolate(blend_choice.permute(0, 3, 1, 2).contiguous(), img_orig.shape[-1], mode='nearest')

    blend_choice = blend_choice * 0.7 + img_orig.cpu() * 0.3

    torchvision.utils.save_image(img_orig.detach().cpu(), os.path.join(output_dir, "final_result.jpg"), 
        nrow=img_orig.shape[0], normalize=True, scale_each=True, range=(-1, 1))
    torchvision.utils.save_image(blend_choice.detach().cpu(), os.path.join(output_dir, "final_cluster.jpg"), 
        nrow=blend_choice.shape[0], normalize=True, scale_each=True, range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description_dir", type=str, default="../celeba-caption", help="the corpus of face descriptions")
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the editing/generation")
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel_multiplier")
    parser.add_argument("--attention_layer", type=int, default=8, help="blende attention layer")
    parser.add_argument("--cluster_num", type=int, default=10, help="cluster num")
    parser.add_argument("--batch_size", type=int, default=1, help="traning batchsize")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda_ess", type=float, default=0.6)
    parser.add_argument("--lambda_sec", type=float, default=0.6)
    parser.add_argument("--lambda_id", type=float, default=0.3)
    parser.add_argument("--lambda_delta", type=float, default=0.008)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                      "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument('--work_in_stylespace', default=False, action='store_true')
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--ir_se50_weights', default='../pretrained_models/model_ir_se50.pth', type=str,
                             help="Path to facial recognition network used in ID loss")
    parser.add_argument('--amp', action='store_true',
                    help='AMP using')

    # dist
    parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('--seed', default=200, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids to use')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

    args = parser.parse_args()

    main(args)