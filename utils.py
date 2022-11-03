import os
import re
# from tkinter import Image
import torch
import clip
import random
import torch_fidelity
from torch_fidelity import calculate_metrics
import torchvision
from torchvision.utils import save_image
import shutil
import math
import json
import argparse
import torch.distributed as dist
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from models.encoders.psp_encoders import Encoder4Editing
from sklearn.metrics import jaccard_score


google_drive_paths = {
    "stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",

    "mapper/pretrained/afro.pt": "https://drive.google.com/uc?id=1i5vAqo4z0I-Yon3FNft_YZOq7ClWayQJ",
    "mapper/pretrained/angry.pt": "https://drive.google.com/uc?id=1g82HEH0jFDrcbCtn3M22gesWKfzWV_ma",
    "mapper/pretrained/beyonce.pt": "https://drive.google.com/uc?id=1KJTc-h02LXs4zqCyo7pzCp0iWeO6T9fz",
    "mapper/pretrained/bobcut.pt": "https://drive.google.com/uc?id=1IvyqjZzKS-vNdq_OhwapAcwrxgLAY8UF",
    "mapper/pretrained/bowlcut.pt": "https://drive.google.com/uc?id=1xwdxI2YCewSt05dEHgkpmmzoauPjEnnZ",
    "mapper/pretrained/curly_hair.pt": "https://drive.google.com/uc?id=1xZ7fFB12Ci6rUbUfaHPpo44xUFzpWQ6M",
    "mapper/pretrained/depp.pt": "https://drive.google.com/uc?id=1FPiJkvFPG_y-bFanxLLP91wUKuy-l3IV",
    "mapper/pretrained/hilary_clinton.pt": "https://drive.google.com/uc?id=1X7U2zj2lt0KFifIsTfOOzVZXqYyCWVll",
    "mapper/pretrained/mohawk.pt": "https://drive.google.com/uc?id=1oMMPc8iQZ7dhyWavZ7VNWLwzf9aX4C09",
    "mapper/pretrained/purple_hair.pt": "https://drive.google.com/uc?id=14H0CGXWxePrrKIYmZnDD2Ccs65EEww75",
    "mapper/pretrained/surprised.pt": "https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI",
    "mapper/pretrained/taylor_swift.pt": "https://drive.google.com/uc?id=10jHuHsKKJxuf3N0vgQbX_SMEQgFHDrZa",
    "mapper/pretrained/trump.pt": "https://drive.google.com/uc?id=14v8D0uzy4tOyfBU3ca9T0AzTt3v-dNyh",
    "mapper/pretrained/zuckerberg.pt": "https://drive.google.com/uc?id=1NjDcMUL8G-pO3i_9N6EPpQNXeMc3Ar1r",

    "example_celebs.pt": "https://drive.google.com/uc?id=1VL3lP4avRhz75LxSza6jgDe-pHd2veQG"
}


def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )


def descripition_corpus(args):
    phras_celeba = []
    sentence_celeba = []
    for root, _, files in os.walk('../celeba-caption'):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                choose_line = torch.randint(10, (2, ))
                all_lines = f.readlines()
                for line in [all_lines[id_line] for id_line in choose_line]:
                    sentence_celeba.extend([line.rstrip('\n')])
                    line = re.split('[,.]', line.rstrip('\n'))[:-1]
                    line = [phra[5:] if phra.startswith(' and') else phra.lstrip(' ') for phra in line]
                    phras_celeba.extend(line)
            # if len(phras) >= 10000:
            #     break

    phras_face2text = []
    sentence_face2text = []
    with open("../face2text_v1.0/raw.json",'r') as f:
        load_dict = json.load(f)

    for line in load_dict:
        sentence_face2text.extend([line['description'].rstrip('\n')])
        line = re.split('[,.]', line['description'].rstrip('\n'))[:-1]
        line = [phra[5:] if phra.startswith(' and') else phra.lstrip(' ') for phra in line]
        phras_face2text.extend(line)

    phras_own = []
    with open(args.own_description_dir,'r') as f:
        all_lines = f.readlines()
    phras_own.extend([line.rstrip('\n') for line in all_lines])
    
    return phras_celeba, phras_face2text, phras_own, sentence_celeba, sentence_face2text


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 


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


class CA_NET(torch.nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, t_dim, c_dim):
        super(CA_NET, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.fc = torch.nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class GLU(torch.nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


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


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.av_pool = torch.nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = torch.nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.clamp_with_grad = ClampWithGrad.apply
        self.augs = transforms.Compose([
            # transforms.RandomApply([transforms.RandomResizedCrop((self.cut_size, self.cut_size), scale=(0.1,1))], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), fillcolor=None)], p=0.8),
            transforms.RandomPerspective(0.2,p=0.4, fill=None),
            transforms.RandomApply([transforms.ColorJitter(hue=0.01, saturation=0.01)], p=0.7), ]
        )
        self.use_augs = True
        self.noise_fac = 0.1

    def set_cut_pow(self, cut_pow):
      self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size/min_size_width)
        
        for ii in range(self.cutn):
          size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)

          offsetx = torch.randint(0, sideX - size + 1, ())
          offsety = torch.randint(0, sideY - size + 1, ())
          cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
          cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)

        if self.use_augs:
          cutouts = self.augs(cutouts)

        if self.noise_fac:
          facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
          cutouts = cutouts + facs * torch.randn_like(cutouts)
        
        return self.clamp_with_grad(cutouts, 0, 1)


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = torch.nn.functional.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = torch.nn.functional.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = torch.nn.functional.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = torch.nn.functional.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return torch.nn.functional.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


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


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]
    masks = masks.squeeze(1)

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        x, y = torch.where(mask > 0.7)
        if x.numel() == 0:
            bounding_boxes[index, 0] = 0
            bounding_boxes[index, 2] = masks.shape[1]-1
        else:
            bounding_boxes[index, 0] = max(torch.min(x) - masks.shape[1]//16, 0)
            bounding_boxes[index, 2] = min(torch.max(x) + masks.shape[1]//16, masks.shape[1]-1)
        if y.numel() == 0:
            bounding_boxes[index, 1] = 0
            bounding_boxes[index, 3] = masks.shape[2]-1
        else:
            bounding_boxes[index, 1] = max(torch.min(y) - masks.shape[2]//16, 0)
            bounding_boxes[index, 3] = min(torch.max(y) + masks.shape[2]//16, masks.shape[2]-1)

    return bounding_boxes.type(torch.int)


def generate_imgs(args, attention_layer, fake_dir, real_dir, phras, iteration, batch, g_ema, Mapper_module, clip_model, id_loss):
    g_ema.eval()
    mean_latent = g_ema.mean_latent(4096)
    Mapper_module.eval()

    upsample = torch.nn.Upsample(scale_factor=7)
    avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)

    img_counter = 0
    img_counter2 = 0
    identity_cos = 0
    improve = 0

    for i in range(iteration):
        if args.latent_path:
            latent_code_init_load = torch.load(args.latent_path).cuda()
            code_choose = torch.randint(len(latent_code_init_load), (batch, )).cuda()
            latent_code_init = latent_code_init_load[code_choose]
        else:
            latent_code_init_not_trunc = torch.randn(batch, 512).cuda()
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)

        if args.work_in_stylespace:
            _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
            latent = [s.detach().clone() for s in latent_code_init]
        else:
            latent = latent_code_init.detach().clone()

        img_orig, _, _, feature_map = g_ema([latent], input_is_latent=True, randomize_noise=False, return_features=True, input_is_stylespace=args.work_in_stylespace)
        feature_map.append(g_ema.input.input.repeat(batch, 1, 1, 1))
        blend_feature = feature_map[attention_layer - 1]
        blend_size = blend_feature.shape[-1]
        image_orig = avg_pool(upsample(img_orig))
        image_features_origin = clip_model.encode_image(image_orig)

        phras_choose = torch.randint(len(phras), (batch, ))
        text = clip.tokenize([phras[choose] for choose in phras_choose], truncate=True).cuda()
        text_features = clip_model.encode_text(text)
        if not args.work_in_stylespace:
            delta_zs, attention_map, _ = Mapper_module.forward(torch.cat([text_features.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1), feature_map, blend_size)
            strength = torch.ones_like(delta_zs)
            # strength[:, w_code_num[attention_layer]:, :] = 0.0
            new_latent_code = latent + strength * delta_zs
        else:
            new_latent_code, attention_map, _ = Mapper_module.forward([torch.cat([text_features.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in latent], feature_map, blend_size)
        
        # attention_map[attention_map<0.8] = 0.0
        img_gen, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace, attention_layer=attention_layer, attention_map=attention_map, feature_map=feature_map)
        image_gen = avg_pool(upsample(img_gen))
        image_features_gen = clip_model.encode_image(image_gen)

        identity_cos += (1 - id_loss(img_gen, img_orig)[0]) * img_gen.shape[0]

        sim_ori = torch.cosine_similarity(image_features_origin, text_features[:, :])
        sim_gen = torch.cosine_similarity(image_features_gen, text_features[:, :])
        improve += torch.sum(sim_gen > (sim_ori))

        for img in img_gen:
            save_image(img,
                        os.path.join(fake_dir, f'{args.rank:0>2}_{img_counter:0>5}.jpg'),
                        normalize=True,
                        range=(-1, 1))
            img_counter += 1
        for img in img_orig:
            save_image(img,
                        os.path.join(real_dir, f'{args.rank:0>2}_{img_counter2:0>5}.jpg'),
                        normalize=True,
                        range=(-1, 1))
            img_counter2 += 1
    return img_counter, identity_cos, improve


def real_imgs(dataset_dir, real_dir, img_counter):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            shutil.copy(os.path.join(root, file), real_dir)
            img_counter -= 1
            if img_counter <= 0:
                break


def cal_evaluation(args, attention_layer, output_dir, phras, g_ema, Mapper_module, clip_model, id_loss, seed=None, iteration=100, batch=1, dataset_dir='../data/CelebAMask-HQ/CelebA-HQ-img', one_gpu=True):
    if seed is not None:
        set_random_seed(seed)
    fake_dir = os.path.join(output_dir, './generate_imgs')
    real_dir = os.path.join(output_dir, './real_imgs')
    if args.rank == 0:
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)
    if not one_gpu:
        dist.barrier()
    with torch.no_grad():
        img_counter, identity_cos, improve = generate_imgs(args, attention_layer, fake_dir, real_dir, phras, iteration, batch, g_ema, Mapper_module, clip_model, id_loss)
    if not one_gpu:
        img_counter = torch.tensor(img_counter).cuda()
        dist.reduce(img_counter, 0)
        img_counter = int(img_counter)
        dist.reduce(identity_cos, 0)
        dist.reduce(improve, 0)
    if args.rank == 0:
        # real_imgs(dataset_dir, real_dir, img_counter)

        metrics_dict = calculate_metrics(input1=fake_dir,
                                        input2=real_dir,
                                        cuda=True,
                                        isc=True,
                                        fid=True,
                                        kid=False,
                                        verbose=False)
        shutil.rmtree(fake_dir)
        shutil.rmtree(real_dir)

        IS = metrics_dict[torch_fidelity.KEY_METRIC_ISC_MEAN]
        FID = metrics_dict[torch_fidelity.KEY_METRIC_FID]
        return IS, FID, identity_cos.item() / float(img_counter), improve.item() / float(img_counter)
    else:
        return 0, 0, 0, 0


class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()
        
        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            label_path = os.path.join(self.label_path, str(i)+'.png')
            # print (img_path, label_path) 
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

    def __getitem__(self, index):
        
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def transform_img(size, centercrop, resize, totensor, normalize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(size))
    if resize:
        options.append(transforms.Resize((size,size)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    return transform


def transform_label(size, centercrop, resize, totensor, normalize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(size))
    if resize:
        options.append(transforms.Resize((size,size)))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
    transform = transforms.Compose(options)
    return transform


def load_e4e_standalone(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = argparse.Namespace(**ckpt['opts'])
    e4e = Encoder4Editing(50, 'ir_se', opts)
    e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    e4e.load_state_dict(e4e_dict)
    e4e.eval()
    e4e = e4e.to(device)
    latent_avg = ckpt['latent_avg'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

    e4e.register_forward_hook(add_latent_avg)
    return e4e


def attention_with_text(text_features, latent, Mapper, attention_layer, work_in_stylespace, feature_map=None):
    blend_feature = feature_map[attention_layer - 1]
    blend_size = blend_feature.shape[-1]

    if not work_in_stylespace:
        _, attention_map, _ = Mapper(torch.cat([text_features.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1), feature_map, blend_size)
    else:
        _, attention_map, _ = Mapper([torch.cat([text_features.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in latent], feature_map, blend_size)

    attention_map = attention_map.view(text_features.shape[0], 1, blend_size, blend_size)
    attention_map[attention_map<0.8] = 0
    attention_map[attention_map>0.7] = 1
    return attention_map


def calculate_IOU(args, attention_layer, blend_size, g_ema, Mapper_module, clip_model):
    e4e_path = '../pretrained_models/e4e_ffhq_encode.pt'
    ckpt = torch.load(e4e_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = e4e_path
    opts = argparse.Namespace(**opts)
    net_e4e = load_e4e_standalone(e4e_path)
    net_e4e.eval()
    net_e4e = net_e4e.cuda()
    g_ema.eval()
    Mapper_module.eval()

    img_path = '../face_parsing/Data_preprocessing/test_img'
    label_path = '../face_parsing/Data_preprocessing/test_label'
    batch_size = 1
    dataset = CelebAMaskHQ(img_path, label_path, transform_img(256, False, True, True, True), transform_label(blend_size, False, True, True, False), mode=True)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=False)

    sequence_text = ["rosy cheeks", "big nose", "brown eyes", "bushy eyebrows", "large ears", "mouths are slightly open", "pink lipsticks", "blonde hair"]
    text_features = []
    with torch.no_grad():
        for text in sequence_text:
            text_inputs = clip.tokenize([text]*batch_size).cuda()
            text_features.append(clip_model.encode_text(text_inputs))
    real_labels = []
    predict_labels = []
    for i, (img, label) in enumerate(loader):
        if i == 90:
            break
        with torch.no_grad():
            latents =  net_e4e(img.cuda())
            _, latents, styles = g_ema([latents], input_is_latent=True, return_latents=True)
            if args.work_in_stylespace:
                latent_code_init = styles
            else:
                latent_code_init = latents
            _, _, _, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True, input_is_stylespace=args.work_in_stylespace)
            feature_map.append(g_ema.input.input.repeat(batch_size, 1, 1, 1))

            predict_label_onetext = []
            for j in range(len(text_features)):
                predict_label_onetext.append(attention_with_text(text_features[j], latent_code_init, Mapper_module, attention_layer, args.work_in_stylespace, feature_map))
            predict_label = torch.cat(predict_label_onetext, dim=1)
            label = (label*255).type(torch.int)
            real_label = torch.zeros_like(label)
            real_label[label==1] = 1
            real_label[label==2] = 2
            real_label[label==4] = 3
            real_label[label==5] = 3
            real_label[label==6] = 4
            real_label[label==7] = 4
            real_label[label==8] = 5
            real_label[label==9] = 5
            real_label[label==10] = 6
            real_label[label==11] = 7
            real_label[label==12] = 7
            real_label[label==13] = 8
            real_map = torch.zeros((predict_label.shape[0], predict_label.shape[1]+1, predict_label.shape[2], predict_label.shape[3]))
            real_label = real_map.scatter_(1, real_label.type(torch.int64), 1)[:, 1:, :, :]
            real_labels.append(real_label)
            predict_labels.append(predict_label)

    real_labels = torch.cat(real_labels, dim=0).permute(0, 2, 3, 1).contiguous().view(-1, real_label.shape[1])
    predict_labels = torch.cat(predict_labels, dim=0).permute(0, 2, 3, 1).contiguous().view(-1, real_label.shape[1])
    each_IOU = jaccard_score(real_labels.numpy(), predict_labels.cpu().numpy(), average=None)
    mean_IOU = jaccard_score(real_labels.numpy(), predict_labels.cpu().numpy(), average='macro')
    print(each_IOU)
    return mean_IOU
