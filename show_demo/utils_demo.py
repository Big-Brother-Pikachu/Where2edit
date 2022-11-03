import torch
import torchvision.transforms as transforms
import argparse
from PIL import Image

from attention.attention_model import Generator, StyledConv, ModulatedConv2d, PixelNorm, EqualLinear, EqualConv2d
from utils import ensure_checkpoint_exists, descripition_corpus, set_random_seed, pairwise_distance, CA_NET
from models.encoders.psp_encoders import Encoder4Editing


class FullSpaceMapperSpatialLin_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512, attention_layer=11, cluster_layer=11, channel_multiplier=1, clusters=10, cluster_dim=512):
        super(FullSpaceMapperSpatialLin_Net, self).__init__()

        total_layers = layers + int((layers - 2) * 0.5)
        dim = [512] * 12 + [256*channel_multiplier] * 3 + [128*channel_multiplier] * 3 + [64*channel_multiplier] * 3 + [32*channel_multiplier] * 3 + [16*channel_multiplier] * 3
        self.layer_num = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
        style_layers = [0, 2, 2, 3, 5, 5, 6, 8, 8, 9, 11, 11, 12, 14, 14, 15, 17, 17, 18, 20, 20, 21, 23, 23, 24, 26, 26]
        self.mapper_layer = style_layers[attention_layer]
        # self.mapper_layer = attention_layer
        for c in range(total_layers):
            if c < self.mapper_layer:
                setattr(self, f"mapper_{c}", EqualLinear(dim[c], dim[c], bias_init=1))
                setattr(self, f"mapper_textca_{c}", CA_NET(latent_dim, latent_dim))
                setattr(self, f"mapper_text_{c}", torch.nn.Sequential(EqualLinear(latent_dim, (latent_dim+512)//2, lr_mul=1, activation='fused_lrelu'), 
                                                                                                                              EqualLinear((latent_dim+512)//2, 512, lr_mul=1, activation='fused_lrelu')))
                setattr(self, f"mapper_all_{c}", EqualLinear(dim[c]+512, dim[c], bias_init=1))
            if c in self.layer_num:
                setattr(self, f"attention_textca_{c}", EqualLinear(latent_dim, dim[c+1], bias_init=1))
                setattr(self, f"attention_{c}", StyledConv(dim[c+1], 32, 1, dim[c+1], blur_kernel=[1, 3, 3, 1]))

        self.attention_textca_first = EqualLinear(latent_dim, dim[0], bias_init=1)
        self.attention_first = StyledConv(dim[0], 32, 1, dim[0], blur_kernel=[1, 3, 3, 1])
        self.attention_textca_last = EqualLinear(latent_dim, 32*layers, bias_init=1)
        self.attention_last = StyledConv(32*layers, 1, 1, 32*layers, blur_kernel=[1, 3, 3, 1])
        self.initial_bias = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.initial_bias, 5)
        self.latent_dim = latent_dim
        self.register_buffer('initial_state', torch.randn(clusters, cluster_dim))
        self.cluster_layer = cluster_layer
        self.clusters = clusters

    def store_clusters(self, initial_state):
        device = self.attention_first.conv.weight.device
        self.initial_state = initial_state.to(device)

    def forward(self, x, feature_map, size, strength_alpha, attention_text=None, mode=3):
        batch = x[0].shape[0]
        initial_state = self.initial_state
        x_text = x[0][:, 0, :self.latent_dim]
        if attention_text is None:
            attention_text = x_text

        with torch.no_grad():
            blend_feature = feature_map[self.cluster_layer - 1]
            cluster_size = blend_feature.shape[2]
            position_channel = blend_feature.shape[1] // 16
            x_position = torch.arange(cluster_size).to(blend_feature.device).float().unsqueeze(0).repeat(cluster_size, 1) * 2 / float(cluster_size-1) - 1
            y_position = torch.arange(cluster_size).to(blend_feature.device).float().unsqueeze(1).repeat(1, cluster_size) * 2 / float(cluster_size-1) - 1
            x_position = x_position.unsqueeze(0).unsqueeze(0).repeat(batch, position_channel, 1, 1)
            y_position = y_position.unsqueeze(0).unsqueeze(0).repeat(batch, position_channel, 1, 1)
            concat_feature = [blend_feature]
            concat_feature.extend([x_position, y_position])
            concat_feature = torch.cat(concat_feature, dim=1)
            channel_nums = concat_feature.shape[1]
            concat_feature = concat_feature.permute(0, 2, 3, 1).contiguous()
            concat_feature = concat_feature.view(-1, channel_nums)
            dis = pairwise_distance(concat_feature, initial_state)
            choice_cluster = torch.arange(batch).to(blend_feature.device).unsqueeze(1).repeat(1, cluster_size**2).view(batch, cluster_size, cluster_size) * self.clusters + torch.argmin(dis, dim=1).view(batch, cluster_size, cluster_size)
            choice_cluster = torch.nn.functional.interpolate(choice_cluster.unsqueeze(1).float(), size).squeeze(1).long()

        out = []
        feature = feature_map[-1]
        x_text_ca = self.attention_textca_first(attention_text)
        feature_res, _ = self.attention_first(feature, x_text_ca.view(batch, 1, -1, 1, 1), input_is_stylespace=True)
        feature_res = torch.nn.functional.interpolate(feature_res, size)
        attention_feature = [feature_res]
        loss_delta = 0
        loss_kl = 0
        for c in range(len(x)):
            if c < self.mapper_layer:
                curr_mapper = getattr(self, f"mapper_text_{c}")
                x_text_hidden = curr_mapper(x_text).unsqueeze(1)
                x_c = x[c][:, :, self.latent_dim:]
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_hidden = curr_mapper(x_c)
                curr_mapper = getattr(self, f"mapper_all_{c}")
                x_c_new = x_c + strength_alpha * (curr_mapper(torch.cat([x_c_hidden, x_text_hidden], dim=-1)) - x_c)
                loss_delta += torch.mean(torch.norm(x_c_new - x_c, dim=-1)) / float(self.mapper_layer)
                out.append(x_c_new.unsqueeze(3).unsqueeze(3))
            else:
                x_c = x[c][:, :, self.latent_dim:]
                out.append(x_c.unsqueeze(3).unsqueeze(3))
            if c in self.layer_num:
                curr_mapper = getattr(self, f"attention_textca_{c}")
                x_text_ca = curr_mapper(attention_text)
                feature = feature_map[c]
                curr_mapper = getattr(self, f"attention_{c}")
                feature_res, _ = curr_mapper(feature, x_text_ca.view(batch, 1, -1, 1, 1), input_is_stylespace=True)
                feature_res = torch.nn.functional.interpolate(feature_res, size)
                attention_feature.append(feature_res)
        
        each_attention_map = torch.cat(attention_feature, dim=1)
        x_text_ca = self.attention_textca_last(attention_text)
        each_attention_map, _ = self.attention_last(each_attention_map, x_text_ca.view(batch, 1, -1, 1, 1), input_is_stylespace=True)
        each_attention_map = torch.nn.Sigmoid()(each_attention_map + self.initial_bias).view(batch, size, size)
        # if self.training:
        attention_map = torch.ones((batch, size, size)).to(blend_feature.device)
        cluster_attention = torch.tensor([0.0]).to(loss_delta.device)
        batch_attention = torch.tensor([0.0]).to(loss_delta.device)
        for i in range(batch * self.clusters):
            same_attention = torch.mean(each_attention_map[choice_cluster==i])
            attention_map[choice_cluster==i] = same_attention
            if not torch.isnan(same_attention):
                # if same_attention > 0.8:
                # cluster_attention += same_attention
                cluster_attention += torch.relu(same_attention - 0.7)
            if (i + 1) % self.clusters == 0:
                # if cluster_attention > 2:
                batch_attention += cluster_attention
                cluster_attention = torch.tensor([0.0]).to(loss_delta.device)
        if mode == 3:
            attention_map = attention_map.unsqueeze(1)
        else:
            attention_map = each_attention_map.unsqueeze(1)
        loss_reg = batch_attention / float(batch)
        # else:
        #     attention_map = each_attention_map.unsqueeze(1)
        #     loss_reg = torch.tensor([0.0]).to(loss_delta.device)

        loss_tv = torch.nn.MSELoss()(each_attention_map.unsqueeze(1), attention_map.detach())
        loss_delta += loss_kl

        final_attention_map = attention_map.clone()
        # final_attention_map[attention_map < 0.8] = attention_map[attention_map < 0.8] - attention_map[attention_map < 0.8].detach()
        # final_attention_map = torchvision.transforms.functional.gaussian_blur(final_attention_map, 5)

        return out, final_attention_map, [loss_delta, loss_reg, loss_tv]


def one_text_edit(text_features, attention_text_features, latent, g_ema, Mapper, attention_layer, work_in_stylespace, strength_alpha, attention_threshold, feature_map=None):
    blend_feature = feature_map[attention_layer - 1]
    blend_size = blend_feature.shape[-1]

    if not work_in_stylespace:
        delta_zs, attention_map, _ = Mapper(torch.cat([text_features.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1), feature_map, blend_size, strength_alpha, attention_text=attention_text_features)
        strength = torch.ones_like(delta_zs)
        # strength[:, w_code_num[attention_layer]:, :] = 0.0
        new_latent_code = latent + strength * delta_zs
    else:
        new_latent_code, attention_map, _ = Mapper([torch.cat([text_features.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in latent], feature_map, blend_size, strength_alpha, attention_text=attention_text_features)

    attention_map[attention_map<attention_threshold] = 0.0
    attention_map = transforms.functional.gaussian_blur(attention_map, 5)
    img_gen, _, _, feature_map = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, return_features=True, input_is_stylespace=work_in_stylespace, attention_layer=attention_layer, attention_map=attention_map, feature_map=feature_map)
    return img_gen, new_latent_code, attention_map, feature_map


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