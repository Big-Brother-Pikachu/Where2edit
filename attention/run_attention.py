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

from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from criteria.perceptual_loss import PerceptualLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from attention_model import Generator, ModulatedConv2d, StyledConv, PixelNorm, EqualLinear, EqualConv2d
import clip
from utils import ensure_checkpoint_exists, descripition_corpus, set_random_seed, cal_evaluation, pairwise_distance, GatherLayer, Logger, Gumbel_softmax, Addnoise, CA_NET, MakeCutouts, masks_to_boxes, calculate_IOU


STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


class Mapper_Net(torch.nn.Module):
    def __init__(self, in_dim=512, latent_dim=512):
        super(Mapper_Net, self).__init__()

        layers = [PixelNorm(dim=2)]

        for i in range(4):
            if i == 0:
                layers.append(
                    EqualLinear(
                        in_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )

        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x


class MapperCon_Net(torch.nn.Module):
    def __init__(self, in_dim=512, latent_dim=512):
        super(MapperCon_Net, self).__init__()

        layers = [PixelNorm(dim=2)]

        for i in range(2):
            if i == 0:
                layers.append(
                    EqualLinear(
                        in_dim - latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )

        self.mapping_text = torch.nn.Sequential(*layers)

        layers = [PixelNorm(dim=2)]

        for i in range(2):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping_latent = torch.nn.Sequential(*layers)

        layers = []

        for i in range(2):
            if i == 0:
                layers.append(
                    EqualLinear(
                        2 * latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )

        self.mapping_together = torch.nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_text = x[:, :, :-self.latent_dim]
        x_latent = x[:, :, -self.latent_dim:]
        x_text = self.mapping_text(x_text)
        x_latent = self.mapping_latent(x_latent)
        x = torch.cat([x_text, x_latent], dim=-1)
        x = self.mapping_together(x)
        return x


class MapperConLin_Net(torch.nn.Module):
    def __init__(self, in_dim=512, latent_dim=512):
        super(MapperConLin_Net, self).__init__()

        layers = [PixelNorm(dim=2)]

        self.mapping_text = torch.nn.Sequential(*layers)

        layers = [PixelNorm(dim=2)]

        self.mapping_latent = torch.nn.Sequential(*layers)

        layers = []
        layers.append(
            EqualLinear(
                in_dim, latent_dim, lr_mul=0.1
            )
        )

        self.mapping_together = torch.nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_text = x[:, :, :-self.latent_dim]
        x_latent = x[:, :, -self.latent_dim:]
        x_text = self.mapping_text(x_text)
        x_latent = self.mapping_latent(x_latent)
        x = torch.cat([x_text, x_latent], dim=-1)
        x = self.mapping_together(x)
        return x


class FullSpaceMapper_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512):
        super(FullSpaceMapper_Net, self).__init__()

        for c in range(layers):
            setattr(self, f"mapper_{c}", Mapper_Net(in_dim=in_dim, latent_dim=latent_dim))

    def forward(self, x):
        out = []
        for c in range(x.shape[1]):
            x_c = x[:, c, :].unsqueeze(1)
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c)
            out.append(x_c_res)

        delta_zs = torch.cat(out, dim=1)
        loss_delta = torch.mean(torch.norm(delta_zs, dim=-1))

        return delta_zs, loss_delta


class FullSpaceMapperCon_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512):
        super(FullSpaceMapperCon_Net, self).__init__()

        for c in range(layers):
            setattr(self, f"mapper_{c}", MapperCon_Net(in_dim=in_dim, latent_dim=latent_dim))

    def forward(self, x):
        out = []
        for c in range(x.shape[1]):
            x_c = x[:, c, :].unsqueeze(1)
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c)
            out.append(x_c_res)
        
        delta_zs = torch.cat(out, dim=1)
        loss_delta = torch.mean(torch.norm(delta_zs, dim=-1))

        return delta_zs, loss_delta


class FullSpaceMapperAtt_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512):
        super(FullSpaceMapperAtt_Net, self).__init__()

        for c in range(layers):
            setattr(self, f"mapper_{c}", MapperCon_Net(in_dim=in_dim, latent_dim=latent_dim))

        attention_layers = [PixelNorm(dim=1)]
        for i in range(2):
            if i == 0:
                attention_layers.append(
                    EqualLinear(
                        in_dim - latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                attention_layers.append(
                    EqualLinear(
                        latent_dim, layers, lr_mul=0.01, activation=None
                    )
                )
        # attention_layers.append(Multiply(4))
        attention_layers.append(Addnoise(0.5))
        attention_layers.append(torch.nn.Sigmoid())
        self.mapping_attention = torch.nn.Sequential(*attention_layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_text = x[:, 0, :-self.latent_dim]
        attention = self.mapping_attention(x_text)
        out = []
        for c in range(x.shape[1]):
            x_c = x[:, c, :].unsqueeze(1)
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c)
            x_c_res = x_c_res * attention[:, c].unsqueeze(1).unsqueeze(1).repeat(1, 1, x_c_res.shape[2])
            out.append(x_c_res)
        
        delta_zs = torch.cat(out, dim=1)
        # loss_delta = torch.mean(torch.norm(delta_zs, dim=-1))
        # loss_att = torch.mean(torch.norm(attention, p=1, dim=-1))
        loss_delta = 0
        loss_att = 0.25 - torch.mean((attention - 0.5) ** 2)

        return delta_zs, loss_delta + loss_att


class FullSpaceMapperAttLin_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512):
        super(FullSpaceMapperAttLin_Net, self).__init__()

        for c in range(layers):
            setattr(self, f"mapper_{c}", MapperConLin_Net(in_dim=in_dim, latent_dim=latent_dim))

        attention_layers = [PixelNorm(dim=1)]
        attention_layers.append(
            EqualLinear(
                in_dim - latent_dim, layers, lr_mul=1
            )
        )
        # attention_layers.append(Multiply(4))
        attention_layers.append(torch.nn.ReLU())
        attention_layers.append(Gumbel_softmax(1))
        # attention_layers.append(Addnoise(0.5))
        # attention_layers.append(torch.nn.Sigmoid())
        self.mapping_attention = torch.nn.Sequential(*attention_layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_text = x[:, 0, :-self.latent_dim]
        attention = self.mapping_attention(x_text)
        out = []
        for c in range(x.shape[1]):
            x_c = x[:, c, :].unsqueeze(1)
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c)
            out.append(x_c_res)
        
        delta_zs = torch.cat(out, dim=1)
        loss_delta = torch.mean(torch.norm(delta_zs, dim=-1))
        delta_zs = delta_zs * attention.unsqueeze(2).repeat(1, 1, delta_zs.shape[2])
        # loss_att = torch.mean(torch.norm(attention, p=1, dim=-1))
        # loss_delta = 0
        # loss_att = 0.25 - torch.mean((attention - 0.5) ** 2)

        return delta_zs, loss_delta + 0


class FullSpaceMapperSpatialLin_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512):
        super(FullSpaceMapperSpatialLin_Net, self).__init__()

        dim = [512] * 7 + [256] * 2 + [128] * 2 + [64] * 2
        self.layer_num = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
        for c in range(layers):
            setattr(self, f"mapper_{c}", MapperConLin_Net(in_dim=in_dim, latent_dim=latent_dim))
            if c < layers - 1:
                setattr(self, f"attention_{c}", EqualConv2d(dim[c], 32, 1))
                # setattr(self, f"attention_{c}", StyledConv(dim[c], 32, 1, latent_dim, blur_kernel=[1, 3, 3, 1]))

        attention_layers = [PixelNorm(dim=1)]
        attention_layers.append(
            EqualLinear(
                in_dim - latent_dim, layers, lr_mul=1
            )
        )
        # attention_layers.append(Multiply(4))
        # attention_layers.append(torch.nn.ReLU())
        # attention_layers.append(Gumbel_softmax(1))
        attention_layers.append(Addnoise(0.5))
        attention_layers.append(torch.nn.Sigmoid())
        self.mapping_attention = torch.nn.Sequential(*attention_layers)
        # self.attention_last = StyledConv(32*(layers-1), 1, 1, latent_dim, blur_kernel=[1, 3, 3, 1])
        self.attention_last = EqualConv2d(32*(layers-1), latent_dim, 1)
        self.proj_text = EqualLinear(latent_dim, latent_dim, lr_mul=1)
        self.latent_dim = latent_dim

    def forward(self, x, feature_map, size):
        x_text = x[:, 0, :-self.latent_dim]
        attention = self.mapping_attention(x_text)
        out = []
        attention_feature = []
        for c in range(x.shape[1]):
            x_c = x[:, c, :].unsqueeze(1)
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c)
            out.append(x_c_res)
            if c < x.shape[1] - 1:
                feature = feature_map[self.layer_num[c]]
                curr_mapper = getattr(self, f"attention_{c}")
                feature_res = curr_mapper(feature)
                # feature_res, _ = curr_mapper(feature, x_text)
                feature_res = torch.nn.functional.interpolate(feature_res, size)
                attention_feature.append(feature_res)
        
        delta_zs = torch.cat(out, dim=1)
        loss_delta = torch.mean(torch.norm(delta_zs, dim=-1))
        # delta_zs = delta_zs * attention.unsqueeze(2).repeat(1, 1, delta_zs.shape[2])
        # loss_att = torch.mean(torch.norm(attention, p=1, dim=-1))
        # loss_delta = 0
        # loss_att = 0.25 - torch.mean((attention - 0.5) ** 2)

        # attention_feature.append(x_text.unsqueeze(2).unsqueeze(3).repeat(1, 1, size, size))
        attention_map = torch.cat(attention_feature, dim=1)
        attention_map = torch.nn.functional.normalize(self.attention_last(attention_map), dim=1)
        proj_text = torch.nn.functional.normalize(self.proj_text(x_text), dim=1).unsqueeze(2).unsqueeze(3).repeat(1, 1, size, size)
        attention_map = 0.5 * (torch.sum(proj_text * attention_map, dim=1).unsqueeze(1) + 1)
        # attention_map = torch.nn.Sigmoid()(attention_map)
        # attention_map = self.attention_last(attention_map)
        small = int(size/4)
        big = int(3*size/4)
        weight_map = torch.ones_like(attention_map)
        weight_map[:, :, small:big, small:big] = 0.5
        loss_reg = torch.mean(weight_map * attention_map)
        loss_tv = torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], p=2) + torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], p=2)

        return delta_zs, attention_map, [loss_delta, loss_reg, loss_tv]


class FullSpaceMapperFEATLin_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512, attention_layer=11, channel_multiplier=1):
        super(FullSpaceMapperFEATLin_Net, self).__init__()

        dim = [512] * 7 + [256*channel_multiplier] * 2 + [128*channel_multiplier] * 2 + [64*channel_multiplier] * 2 + [32*channel_multiplier] * 2 + [16*channel_multiplier] * 2
        self.layer_num = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
        w_code_num = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 18]
        self.mapper_layer = w_code_num[attention_layer]
        for c in range(layers):
            if c < self.mapper_layer:
                # setattr(self, f"mapper_{c}", MapperConLin_Net(in_dim=latent_dim, latent_dim=latent_dim))
                setattr(self, f"mapper_{c}", torch.nn.Sequential(PixelNorm(dim=2), 
                                                                                                                   EqualLinear(latent_dim, latent_dim, lr_mul=0.1, activation='fused_lrelu'), 
                                                                                                                   EqualLinear(latent_dim, latent_dim, lr_mul=0.1, activation='fused_lrelu'), 
                                                                                                                   EqualLinear(latent_dim, latent_dim, lr_mul=0.1, activation='fused_lrelu')))
                # setattr(self, f"mapper_{c}", torch.nn.Sequential(EqualLinear(latent_dim, latent_dim, lr_mul=1, activation='fused_lrelu'), 
                #                                                                                                    EqualLinear(latent_dim, latent_dim, lr_mul=1, activation='fused_lrelu'), 
                #                                                                                                    EqualLinear(latent_dim, latent_dim, lr_mul=1, activation='fused_lrelu')))
            if c < layers - 1:
                # setattr(self, f"attention_{c}", torch.nn.Sequential(EqualConv2d(dim[c], 32, 1), 
                #                                                                                                       torch.nn.ReLU(inplace=True), 
                #                                                                                                       torch.nn.BatchNorm2d(32)))
                setattr(self, f"attention_{c}", EqualConv2d(dim[c], 32, 1))
                # setattr(self, f"attention_{c}", StyledConv(dim[c], 32, 1, latent_dim, blur_kernel=[1, 3, 3, 1]))

        # self.attention_first = torch.nn.Sequential(EqualConv2d(dim[0], 32, 1), 
        #                                                                                       torch.nn.ReLU(inplace=True), 
        #                                                                                       torch.nn.BatchNorm2d(32))
        self.attention_first = EqualConv2d(dim[0], 32, 1)
        # self.attention_last = StyledConv(32*(layers-1), 1, 1, latent_dim, blur_kernel=[1, 3, 3, 1])
        self.attention_last = EqualConv2d(32*layers, 1, 1)
        torch.nn.init.constant_(self.attention_last.bias, 5)
        self.latent_dim = latent_dim

    def forward(self, x, feature_map, size):
        out = []
        feature = feature_map[-1]
        feature_res = self.attention_first(feature)
        feature_res = torch.nn.functional.interpolate(feature_res, size)
        attention_feature = [feature_res]
        for c in range(x.shape[1]):
            if c < self.mapper_layer:
                x_c = x[:, c, -self.latent_dim:].unsqueeze(1)
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c)
                out.append(x_c_res)
            else:
                x_c = x[:, c, -self.latent_dim:].unsqueeze(1)
                x_c_res = torch.zeros_like(x_c)
                out.append(x_c_res)
            if c < x.shape[1] - 1:
                feature = feature_map[self.layer_num[c]]
                curr_mapper = getattr(self, f"attention_{c}")
                feature_res = curr_mapper(feature)
                feature_res = torch.nn.functional.interpolate(feature_res, size)
                attention_feature.append(feature_res)
        
        delta_zs = torch.cat(out, dim=1)
        loss_delta = torch.mean(torch.norm(delta_zs[:, :self.mapper_layer], dim=-1))
        attention_map = torch.cat(attention_feature, dim=1)
        attention_map = self.attention_last(attention_map)
        attention_map = torch.nn.Sigmoid()(attention_map)
        loss_tv = torch.mean(torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], dim=[2,3]) / float((attention_map.shape[2]-1)*attention_map.shape[3]) + 
                                                     torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], dim=[2,3]) / float(attention_map.shape[2]*(attention_map.shape[3]-1)))
        small = int(size/4)
        big = int(3*size/4)
        weight_map = torch.ones_like(attention_map)
        # weight_map[:, :, small:big, small:big] = 0.5

        final_attention_map = attention_map.clone()
        final_attention_map[attention_map < 0.8] = attention_map[attention_map < 0.8] - attention_map[attention_map < 0.8].detach()
        loss_reg = torch.mean(weight_map * final_attention_map)
        
        return delta_zs, final_attention_map, [loss_delta, loss_reg, loss_tv]


class FullSpaceMapperFEATClusterLin_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512, attention_layer=11, cluster_layer=11, channel_multiplier=1, clusters=10, cluster_dim=512):
        super(FullSpaceMapperFEATClusterLin_Net, self).__init__()

        dim = [512] * 7 + [256*channel_multiplier] * 2 + [128*channel_multiplier] * 2 + [64*channel_multiplier] * 2 + [32*channel_multiplier] * 2 + [16*channel_multiplier] * 2
        self.layer_num = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
        w_code_num = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 18]
        self.mapper_layer = w_code_num[attention_layer]
        for c in range(layers):
            if c < self.mapper_layer:
                # setattr(self, f"mapper_{c}", MapperConLin_Net(in_dim=latent_dim, latent_dim=latent_dim))
                setattr(self, f"mapper_{c}", torch.nn.Sequential(PixelNorm(dim=2), 
                                                                                                                   EqualLinear(latent_dim, latent_dim, lr_mul=0.1, activation='fused_lrelu'), 
                                                                                                                   EqualLinear(latent_dim, latent_dim, lr_mul=0.1, activation='fused_lrelu'), 
                                                                                                                   EqualLinear(latent_dim, latent_dim, lr_mul=0.1, activation='fused_lrelu')))
            if c < layers - 1:
                # setattr(self, f"attention_{c}", torch.nn.Sequential(EqualConv2d(dim[c], 32, 1), 
                #                                                                                                       torch.nn.ReLU(inplace=True), 
                #                                                                                                       torch.nn.BatchNorm2d(32)))
                setattr(self, f"attention_{c}", EqualConv2d(dim[c], 32, 1))
                # setattr(self, f"attention_{c}", StyledConv(dim[c], 32, 1, latent_dim, blur_kernel=[1, 3, 3, 1]))

        # self.attention_first = torch.nn.Sequential(EqualConv2d(dim[0], 32, 1), 
        #                                                                                       torch.nn.ReLU(inplace=True), 
        #                                                                                       torch.nn.BatchNorm2d(32))
        self.attention_first = EqualConv2d(dim[0], 32, 1)
        # self.attention_last = StyledConv(32*(layers-1), 1, 1, latent_dim, blur_kernel=[1, 3, 3, 1])
        self.attention_last = EqualConv2d(32*layers, 1, 1)
        torch.nn.init.constant_(self.attention_last.bias, 5)
        # self.attention_cluster = torch.nn.Sequential(PixelNorm(dim=1), 
        #                                                                                            EqualLinear(cluster_dim, cluster_dim, lr_mul=1, activation='fused_lrelu'), 
        #                                                                                            EqualLinear(cluster_dim, 1, lr_mul=0.1), 
        #                                                                                            torch.nn.Sigmoid())
        # torch.nn.init.constant_(self.attention_cluster[2].bias, 50)
        self.latent_dim = latent_dim
        self.register_buffer('initial_state', torch.randn(clusters, cluster_dim))
        self.cluster_layer = cluster_layer
        self.clusters = clusters

    def store_clusters(self, initial_state):
        device = self.attention_first.weight.device
        self.initial_state = initial_state.to(device)

    def forward(self, x, feature_map, size):
        initial_state = self.initial_state
        batch = x.shape[0]
        # attention_value = self.attention_cluster(initial_state).view(1, self.clusters).repeat(batch, 1)

        # final_attention_value = attention_value.clone()
        # final_attention_value[attention_value < 0.8] = attention_value[attention_value < 0.8] - attention_value[attention_value < 0.8].detach()
        # sample_value = torch.sum(final_attention_value, dim=1)
        # sample_value = torch.sum(final_attention_value, dim=1) - 3
        # final_sample_value = sample_value.clone()
        # final_sample_value[sample_value < 0] = sample_value[sample_value < 0] - sample_value[sample_value < 0].detach()

        # loss_reg = torch.mean(final_sample_value)
        # final_attention_value = final_attention_value.view(batch * self.clusters)

        with torch.no_grad():
            blend_feature = feature_map[self.cluster_layer - 1]
            size = blend_feature.shape[2]
            position_channel = blend_feature.shape[1] // 16
            x_position = torch.arange(size).to(blend_feature.device).float().unsqueeze(0).repeat(size, 1) * 2 / float(size-1) - 1
            y_position = torch.arange(size).to(blend_feature.device).float().unsqueeze(1).repeat(1, size) * 2 / float(size-1) - 1
            x_position = x_position.unsqueeze(0).unsqueeze(0).repeat(batch, position_channel, 1, 1)
            y_position = y_position.unsqueeze(0).unsqueeze(0).repeat(batch, position_channel, 1, 1)
            concat_feature = [blend_feature]
            # for j in [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]:
            #     blend_feature = torch.nn.functional.interpolate(feature_map[j].detach().clone(), size)
            #     concat_feature.append(blend_feature)
            concat_feature.extend([x_position, y_position])
            concat_feature = torch.cat(concat_feature, dim=1)
            channel_nums = concat_feature.shape[1]
            concat_feature = concat_feature.permute(0, 2, 3, 1).contiguous()
            concat_feature = concat_feature.view(-1, channel_nums)
            dis = pairwise_distance(concat_feature, initial_state)
            choice_cluster = torch.arange(batch).to(blend_feature.device).unsqueeze(1).repeat(1, size**2).view(batch, size, size) * self.clusters + torch.argmin(dis, dim=1).view(batch, size, size)
        # attention_map = torch.ones((batch, size, size)).to(blend_feature.device)
        # for i in range(batch * self.clusters):
        #     attention_map[choice_cluster==i] = final_attention_value[i]
        # attention_map = attention_map.unsqueeze(1)

        out = []
        feature = feature_map[-1]
        feature_res = self.attention_first(feature)
        feature_res = torch.nn.functional.interpolate(feature_res, size)
        attention_feature = [feature_res]
        for c in range(x.shape[1]):
            if c < self.mapper_layer:
                x_c = x[:, c, -self.latent_dim:].unsqueeze(1)
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c)
                out.append(x_c_res)
            else:
                x_c = x[:, c, -self.latent_dim:].unsqueeze(1)
                x_c_res = torch.zeros_like(x_c)
                out.append(x_c_res)
            if c < x.shape[1] - 1:
                feature = feature_map[self.layer_num[c]]
                curr_mapper = getattr(self, f"attention_{c}")
                feature_res = curr_mapper(feature)
                feature_res = torch.nn.functional.interpolate(feature_res, size)
                attention_feature.append(feature_res)
        
        delta_zs = torch.cat(out, dim=1)
        loss_delta = torch.mean(torch.norm(delta_zs[:, :self.mapper_layer], dim=-1))

        each_attention_map = torch.cat(attention_feature, dim=1)
        each_attention_map = self.attention_last(each_attention_map)
        each_attention_map = torch.nn.Sigmoid()(each_attention_map).view(batch, size, size)
        if self.training:
            attention_map = torch.ones((batch, size, size)).to(blend_feature.device)
            cluster_attention = torch.tensor([0.0]).to(loss_delta.device)
            batch_attention = torch.tensor([0.0]).to(loss_delta.device)
            for i in range(batch * self.clusters):
                same_attention = torch.mean(each_attention_map[choice_cluster==i])
                attention_map[choice_cluster==i] = same_attention
                if not torch.isnan(same_attention):
                    # if same_attention > 0.8:
                    #     cluster_attention += same_attention
                    cluster_attention += torch.relu(same_attention - 0.8)
                if (i + 1) % self.clusters == 0:
                    # if cluster_attention > 2:
                    batch_attention += cluster_attention
                    cluster_attention = torch.tensor([0.0]).to(loss_delta.device)
            attention_map = attention_map.unsqueeze(1)
            loss_reg = batch_attention / float(batch)
        else:
            attention_map = each_attention_map.unsqueeze(1)
            loss_reg = torch.tensor([0.0]).to(loss_delta.device)

        # loss_tv = torch.tensor([0.0]).to(loss_delta.device)
        loss_tv = torch.nn.MSELoss()(each_attention_map.unsqueeze(1), attention_map.detach())
        # loss_tv = torch.mean(torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], dim=[2,3]) / float((attention_map.shape[2]-1)*attention_map.shape[3]) + 
        #                                              torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], dim=[2,3]) / float(attention_map.shape[2]*(attention_map.shape[3]-1)))
        # small = int(size/4)
        # big = int(3*size/4)
        # weight_map = torch.ones_like(attention_map)
        # weight_map[:, :, small:big, small:big] = 0.5

        final_attention_map = attention_map.clone()
        final_attention_map[attention_map < 0.8] = attention_map[attention_map < 0.8] - attention_map[attention_map < 0.8].detach()
        final_attention_map = torchvision.transforms.functional.gaussian_blur(final_attention_map, 5)
        # loss_reg = torch.mean(weight_map * final_attention_map)
        
        return delta_zs, final_attention_map, [loss_delta, loss_reg, loss_tv]


class FullSpaceMapperAttLinStyle_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512):
        super(FullSpaceMapperAttLinStyle_Net, self).__init__()

        total_layers = layers + int((layers - 2) * 0.5)
        dim = [512] * 12 + [256] * 3 + [128] * 3 + [64] * 2
        for c in range(total_layers):
            setattr(self, f"mapper_{c}", MapperConLin_Net(in_dim=in_dim-latent_dim+dim[c], latent_dim=dim[c]))

        attention_layers = [PixelNorm(dim=1)]
        attention_layers.append(
            EqualLinear(
                in_dim - latent_dim, total_layers, lr_mul=1
            )
        )
        # attention_layers.append(Multiply(4))
        # attention_layers.append(torch.nn.ReLU())
        # attention_layers.append(Gumbel_softmax(1))
        attention_layers.append(Addnoise(0.5))
        attention_layers.append(torch.nn.Sigmoid())
        self.mapping_attention = torch.nn.Sequential(*attention_layers)
        self.latent_dim = in_dim - latent_dim

    def forward(self, x):
        x_text = x[0][:, 0, :self.latent_dim]
        attention = self.mapping_attention(x_text)
        out = []
        loss_delta = 0
        for c in range(len(x)):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x[c])
            loss_delta += torch.mean(torch.norm(x_c_res, dim=-1))
            if self.training:
                strength = ((1 + 0.2 * torch.rand((x_c_res.shape[0])).to(x_c_res.device)) * attention[:, c]).unsqueeze(1).unsqueeze(1).repeat(1, *x_c_res.shape[1:])
            else:
                strength = attention[:, c].unsqueeze(1).unsqueeze(1).repeat(1, *x_c_res.shape[1:])
            x_c = x[c][:, :, self.latent_dim:] + strength * x_c_res
            out.append(x_c.view(x_c.shape[0], 1, x_c.shape[2], 1, 1))
            
        # loss_att = torch.mean(torch.norm(attention, p=1, dim=-1))
        # loss_delta = 0
        # loss_att = 0.25 - torch.mean((attention - 0.5) ** 2)

        return out, loss_delta / float(len(x)) + 0


class FullSpaceMapperFEATLinStyle_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512, attention_layer=11, channel_multiplier=1):
        super(FullSpaceMapperFEATLinStyle_Net, self).__init__()

        total_layers = layers + int((layers - 2) * 0.5)
        dim = [512] * 12 + [256*channel_multiplier] * 3 + [128*channel_multiplier] * 3 + [64*channel_multiplier] * 3 + [32*channel_multiplier] * 3 + [16*channel_multiplier] * 3
        self.layer_num = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
        self.mapper_layer = attention_layer
        for c in range(total_layers):
            if c < self.mapper_layer:
                setattr(self, f"mapper_{c}", torch.nn.Sequential(PixelNorm(dim=2), 
                                                                                                                   EqualLinear(dim[c], dim[c], lr_mul=10.0, activation='fused_lrelu'), 
                                                                                                                   EqualLinear(dim[c], dim[c], lr_mul=10.0, activation='fused_lrelu')))
            if c in self.layer_num:
                setattr(self, f"attention_{c}", EqualConv2d(dim[c+1], 32, 1))

        self.attention_last = EqualConv2d(32*(layers-1), 1, 1)
        self.latent_dim = latent_dim

    def forward(self, x, feature_map, size):
        out = []
        attention_feature = []
        loss_delta = 0
        for c in range(len(x)):
            if c < self.mapper_layer:
                x_c = x[c][:, :, self.latent_dim:]
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c)
                loss_delta += torch.mean(torch.norm(x_c_res, dim=-1)) / float(self.mapper_layer)
                strength = torch.ones_like(x_c_res)
                x_c_new = x_c + strength * x_c_res
                out.append(x_c_new.unsqueeze(3).unsqueeze(3))
            else:
                x_c = x[c][:, :, self.latent_dim:]
                out.append(x_c.unsqueeze(3).unsqueeze(3))
            if c in self.layer_num:
                feature = feature_map[c]
                curr_mapper = getattr(self, f"attention_{c}")
                feature_res = curr_mapper(feature)
                feature_res = torch.nn.functional.interpolate(feature_res, size)
                attention_feature.append(feature_res)
        
        attention_map = torch.cat(attention_feature, dim=1)
        attention_map = self.attention_last(attention_map)
        attention_map = torch.nn.Sigmoid()(attention_map)
        small = int(size/4)
        big = int(3*size/4)
        weight_map = torch.ones_like(attention_map)
        # weight_map[:, :, small:big, small:big] = 0.5
        loss_reg = torch.mean(weight_map * attention_map)
        loss_tv = torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], p=2) + torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], p=2)
        # loss_tv = torch.mean(torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], dim=[2,3]) / float((attention_map.shape[2]-1)*attention_map.shape[3]) + 
        #                                              torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], dim=[2,3]) / float(attention_map.shape[2]*(attention_map.shape[3]-1)))

        return out, attention_map, [loss_delta, loss_reg, loss_tv]


class FullSpaceMapperFEATClusterLinStyle_Net(torch.nn.Module):

    def __init__(self, layers, in_dim=512, latent_dim=512, attention_layer=11, cluster_layer=11, channel_multiplier=1, clusters=10, cluster_dim=512):
        super(FullSpaceMapperFEATClusterLinStyle_Net, self).__init__()

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
                # setattr(self, f"mapper_all_{c}", EqualLinear(dim[c]+512, dim[c], lr_mul=0.1, activation='fused_lrelu'))
            if c in self.layer_num:
                # setattr(self, f"attention_{c}", EqualConv2d(dim[c+1], 32, 1))
                setattr(self, f"attention_textca_{c}", EqualLinear(latent_dim, dim[c+1], bias_init=1))
                # setattr(self, f"attention_textca_{c}", CA_NET(latent_dim, dim[c+1]))
                setattr(self, f"attention_{c}", StyledConv(dim[c+1], 32, 1, dim[c+1], blur_kernel=[1, 3, 3, 1]))

        # self.attention_first = EqualConv2d(dim[0], 32, 1)
        self.attention_textca_first = EqualLinear(latent_dim, dim[0], bias_init=1)
        # self.attention_textca_first = CA_NET(latent_dim, dim[0])
        self.attention_first = StyledConv(dim[0], 32, 1, dim[0], blur_kernel=[1, 3, 3, 1])
        # self.attention_last = EqualConv2d(32*layers, 1, 1)
        self.attention_textca_last = EqualLinear(latent_dim, 32*layers, bias_init=1)
        # self.attention_textca_last = CA_NET(latent_dim, 32*layers)
        self.attention_last = StyledConv(32*layers, 1, 1, 32*layers, blur_kernel=[1, 3, 3, 1])
        # torch.nn.init.constant_(self.attention_last.bias, 5)
        self.initial_bias = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.initial_bias, 5)
        # self.attention_text = torch.nn.Sequential(EqualLinear(latent_dim, (latent_dim+512)//2, lr_mul=0.1, activation='fused_lrelu'), 
        #                                                                                      EqualLinear((latent_dim+512)//2, 512, lr_mul=0.1, activation='fused_lrelu'))
        # self.attention_cluster = EqualLinear(cluster_dim, cluster_dim, lr_mul=1, activation='fused_lrelu')
        # self.attention_all = torch.nn.Sequential(EqualLinear(cluster_dim + 512, 1, lr_mul=0.1, bias_init=50), 
        #                                                                                            torch.nn.Sigmoid())
        self.latent_dim = latent_dim
        self.register_buffer('initial_state', torch.randn(clusters, cluster_dim))
        self.cluster_layer = cluster_layer
        self.clusters = clusters

    def store_clusters(self, initial_state):
        device = self.attention_first.conv.weight.device
        assert self.initial_state.shape[0] == initial_state.shape[0], self.initial_state.shape[1] == initial_state.shape[1]
        self.initial_state = initial_state.to(device)

    def forward(self, x, feature_map, size, attention_text=None):
        batch = x[0].shape[0]
        initial_state = self.initial_state
        x_text = x[0][:, 0, :self.latent_dim]
        if attention_text is None:
            attention_text = x_text

        # attention_text = self.attention_text(attention_text).unsqueeze(1).repeat(1, self.clusters, 1)
        # attention_value = self.attention_cluster(initial_state).unsqueeze(0).repeat(batch, 1, 1)
        # attention_value = torch.cat([attention_text, attention_value], dim=-1)
        # attention_value = self.attention_all(attention_value).view(batch, self.clusters)

        # final_attention_value = attention_value.clone()
        # final_attention_value[attention_value < 0.8] = attention_value[attention_value < 0.8] - attention_value[attention_value < 0.8].detach()
        # sample_value = torch.sum(final_attention_value, dim=1) - 3
        # final_sample_value = sample_value.clone()
        # final_sample_value[sample_value < 0] = sample_value[sample_value < 0] - sample_value[sample_value < 0].detach()

        # loss_reg = torch.mean(final_sample_value)
        # final_attention_value = final_attention_value.view(batch * self.clusters)

        with torch.no_grad():
            blend_feature = feature_map[self.cluster_layer - 1]
            cluster_size = blend_feature.shape[2]
            position_channel = blend_feature.shape[1] // 16
            x_position = torch.arange(cluster_size).to(blend_feature.device).float().unsqueeze(0).repeat(cluster_size, 1) * 2 / float(cluster_size-1) - 1
            y_position = torch.arange(cluster_size).to(blend_feature.device).float().unsqueeze(1).repeat(1, cluster_size) * 2 / float(cluster_size-1) - 1
            x_position = x_position.unsqueeze(0).unsqueeze(0).repeat(batch, position_channel, 1, 1)
            y_position = y_position.unsqueeze(0).unsqueeze(0).repeat(batch, position_channel, 1, 1)
            concat_feature = [blend_feature]
            # for j in [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]:
            #     blend_feature = torch.nn.functional.interpolate(feature_map[j].detach().clone(), cluster_size)
            #     concat_feature.append(blend_feature)
            concat_feature.extend([x_position, y_position])
            concat_feature = torch.cat(concat_feature, dim=1)
            channel_nums = concat_feature.shape[1]
            concat_feature = concat_feature.permute(0, 2, 3, 1).contiguous()
            concat_feature = concat_feature.view(-1, channel_nums)
            dis = pairwise_distance(concat_feature, initial_state)
            choice_cluster = torch.arange(batch).to(blend_feature.device).unsqueeze(1).repeat(1, cluster_size**2).view(batch, cluster_size, cluster_size) * self.clusters + torch.argmin(dis, dim=1).view(batch, cluster_size, cluster_size)
            choice_cluster = torch.nn.functional.interpolate(choice_cluster.unsqueeze(1).float(), size).squeeze(1).long()
        # attention_map = torch.ones((batch, cluster_size, cluster_size)).to(blend_feature.device)
        # for i in range(batch * self.clusters):
        #     attention_map[choice_cluster==i] = final_attention_value[i]
        # attention_map = attention_map.unsqueeze(1)

        out = []
        feature = feature_map[-1]
        # feature_res = self.attention_first(feature)
        x_text_ca = self.attention_textca_first(attention_text)
        # x_text_ca, mu, logvar = self.attention_textca_first(attention_text)
        feature_res, _ = self.attention_first(feature, x_text_ca.view(batch, 1, -1, 1, 1), input_is_stylespace=True)
        feature_res = torch.nn.functional.interpolate(feature_res, size)
        attention_feature = [feature_res]
        loss_delta = 0
        loss_kl = 0
        # loss_kl = KL_loss(mu, logvar)
        for c in range(len(x)):
            if c < self.mapper_layer:
                # curr_mapper = getattr(self, f"mapper_textca_{c}")
                # x_text_ca, mu, logvar = curr_mapper(x_text)
                # loss_kl += KL_loss(mu, logvar)
                curr_mapper = getattr(self, f"mapper_text_{c}")
                x_text_hidden = curr_mapper(x_text).unsqueeze(1)
                x_c = x[c][:, :, self.latent_dim:]
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_hidden = curr_mapper(x_c)
                curr_mapper = getattr(self, f"mapper_all_{c}")
                # x_c_new = x_c + curr_mapper(torch.cat([x_c_hidden, x_text_hidden], dim=-1))
                x_c_new = x_c + 0.1 * (curr_mapper(torch.cat([x_c_hidden, x_text_hidden], dim=-1)) - x_c)
                loss_delta += torch.mean(torch.norm(x_c_new - x_c, dim=-1)) / float(self.mapper_layer)
                out.append(x_c_new.unsqueeze(3).unsqueeze(3))
            else:
                x_c = x[c][:, :, self.latent_dim:]
                out.append(x_c.unsqueeze(3).unsqueeze(3))
            if c in self.layer_num:
                curr_mapper = getattr(self, f"attention_textca_{c}")
                x_text_ca = curr_mapper(attention_text)
                # x_text_ca, mu, logvar = curr_mapper(attention_text)
                # loss_kl += KL_loss(mu, logvar)
                feature = feature_map[c]
                curr_mapper = getattr(self, f"attention_{c}")
                # feature_res = curr_mapper(feature)
                feature_res, _ = curr_mapper(feature, x_text_ca.view(batch, 1, -1, 1, 1), input_is_stylespace=True)
                feature_res = torch.nn.functional.interpolate(feature_res, size)
                attention_feature.append(feature_res)
        
        each_attention_map = torch.cat(attention_feature, dim=1)
        x_text_ca = self.attention_textca_last(attention_text)
        # x_text_ca, mu, logvar = self.attention_textca_last(attention_text)
        # loss_kl += KL_loss(mu, logvar)
        each_attention_map, _ = self.attention_last(each_attention_map, x_text_ca.view(batch, 1, -1, 1, 1), input_is_stylespace=True)
        # each_attention_map = self.attention_last(each_attention_map)
        each_attention_map = torch.nn.Sigmoid()(each_attention_map + self.initial_bias).view(batch, size, size)
        # if self.training:
        same_attention_map = torch.ones((batch, size, size)).to(blend_feature.device)
        cluster_attention = torch.tensor([0.0]).to(loss_delta.device)
        batch_attention = torch.tensor([0.0]).to(loss_delta.device)
        loss_tv = torch.tensor([0.0]).to(loss_delta.device)
        # cluster_notnan = 0
        # cluster_std = torch.tensor([0.0]).to(loss_delta.device)
        for i in range(batch * self.clusters):
            same_attention = torch.mean(each_attention_map[choice_cluster==i])
            same_attention_map[choice_cluster==i] = same_attention
            if not torch.isnan(same_attention):
                # cluster_notnan += 1
                # cluster_std += torch.std(each_attention_map[choice_cluster==i], unbiased=False) / float(batch)
            # if not torch.isnan(same_attention):
                # if same_attention > 0.8:
                # cluster_attention += same_attention
                cluster_attention += torch.relu(same_attention - 0.7)
            if (i + 1) % self.clusters == 0:
                # if cluster_attention > 2:
                batch_attention += cluster_attention
                cluster_attention = torch.tensor([0.0]).to(loss_delta.device)
                # loss_tv += cluster_std / float(cluster_notnan)
                # loss_tv += cluster_std
                # cluster_notnan = 0
                # cluster_std = torch.tensor([0.0]).to(loss_delta.device)
        attention_map = same_attention_map.unsqueeze(1)
        loss_reg = batch_attention / float(batch)
        # else:
        #     attention_map = each_attention_map.unsqueeze(1)
        #     loss_reg = torch.tensor([0.0]).to(loss_delta.device)

        loss_tv = torch.nn.MSELoss()(each_attention_map, same_attention_map.detach())
        loss_delta += loss_kl

        final_attention_map = attention_map.clone()
        final_attention_map[attention_map < 0.8] = attention_map[attention_map < 0.8] - attention_map[attention_map < 0.8].detach()
        final_attention_map = torchvision.transforms.functional.gaussian_blur(final_attention_map, 5)

        # weight_map = torch.ones_like(attention_map)
        # weight_map[:, :, small:big, small:big] = 0.5
        # loss_reg = torch.mean(weight_map * attention_map)
        # loss_tv = torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], p=2) + torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], p=2)
        # loss_tv = torch.mean(torch.norm(attention_map[:, :, 1:, :] - attention_map[:, :, :-1, :], dim=[2,3]) / float((attention_map.shape[2]-1)*attention_map.shape[3]) + 
        #                                              torch.norm(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1], dim=[2,3]) / float(attention_map.shape[2]*(attention_map.shape[3]-1)))

        return out, final_attention_map, [loss_delta, loss_reg, loss_tv]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.gpu = None
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
        main_worker(args.gpu, ngpus_per_node, args)


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
        files = ['./run_attention.py', './attention_model.py', '../utils.py']
        for f in files:
            shutil.copy(f, os.path.join(output_dir, f.split('/')[-1]))
        stdout_backup = sys.stdout
        sys.stdout = Logger(stdout_backup, os.path.join(output_dir, 'run.log'))
        print('--------args----------')
        for k in list(vars(args).keys()):    
            print('%s: %s' % (k, vars(args)[k]))
        print('--------args----------\n')

    ensure_checkpoint_exists(args.ckpt)
    phras_celeba, phras_face2text, phras_own, sentence_celeba, sentence_face2text = descripition_corpus(args)
    phras = phras_celeba
    sentence = phras_celeba

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    g_ema = Generator(args.stylegan_size, 512, 8, channel_multiplier=args.channel_multiplier)
    # w_code_num = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13]
    if args.gpu is None:
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        g_ema.load_state_dict(torch.load(args.ckpt, map_location=loc)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda(args.gpu)
    mean_latent = g_ema.mean_latent(4096)

    if args.use_cluster:
        clusters = args.cluster_num
        cluster_layer = args.cluster_layer
        cluster_dim = 512
        initial_state = torch.rand(clusters, cluster_dim)
        if args.cluster_path:
            with open(args.cluster_path, "rb") as f:
                initial_state = pickle.load(f)
                # initial_state = torch.from_numpy(kmeans.cluster_centers_).cuda()
            clusters = initial_state.shape[0]
            # cluster_layer = 13
            cluster_layer = args.cluster_layer
            cluster_dim = initial_state.shape[1]

    clip_loss = CLIPLoss(args)
    id_loss = IDLoss(args, args.gpu)
    perceptual_loss = PerceptualLoss(args)
    make_cutouts = MakeCutouts(clip_loss.model.visual.input_resolution, cutn=args.cutn, cut_pow=1)
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # Mapper = torch.nn.Linear(clip_loss.model.visual.output_dim + mean_latent.shape[-1], mean_latent.shape[-1])
    # Mapper = FullSpaceMapperCon_Net(g_ema.n_latent, clip_loss.model.visual.output_dim + mean_latent.shape[-1], mean_latent.shape[-1])
    # Mapper = FullSpaceMapperAtt_Net(g_ema.n_latent, clip_loss.model.visual.output_dim + mean_latent.shape[-1], mean_latent.shape[-1])
    if not args.work_in_stylespace and not args.use_cluster:
        Mapper = FullSpaceMapperFEATLin_Net(g_ema.n_latent, clip_loss.model.visual.output_dim + mean_latent.shape[-1], mean_latent.shape[-1], attention_layer=args.attention_layer, channel_multiplier=args.channel_multiplier)
    elif not args.work_in_stylespace:
        Mapper = FullSpaceMapperFEATClusterLin_Net(g_ema.n_latent, clip_loss.model.visual.output_dim + mean_latent.shape[-1], mean_latent.shape[-1], attention_layer=args.attention_layer, channel_multiplier=args.channel_multiplier, cluster_layer=cluster_layer, clusters=clusters, cluster_dim=cluster_dim)
    elif args.use_cluster:
        Mapper = FullSpaceMapperFEATClusterLinStyle_Net(g_ema.n_latent, clip_loss.model.visual.output_dim + mean_latent.shape[-1], clip_loss.model.visual.output_dim, attention_layer=args.attention_layer, channel_multiplier=args.channel_multiplier, cluster_layer=cluster_layer, clusters=clusters, cluster_dim=cluster_dim)
    else:
        Mapper = FullSpaceMapperFEATLinStyle_Net(g_ema.n_latent, clip_loss.model.visual.output_dim + mean_latent.shape[-1], clip_loss.model.visual.output_dim, attention_layer=args.attention_layer, channel_multiplier=args.channel_multiplier)

    if args.distributed:
        if args.gpu is not None:
            Mapper.cuda(args.gpu)
            Mapper = torch.nn.parallel.DistributedDataParallel(Mapper, device_ids=[args.gpu], find_unused_parameters=True)
            Mapper_module = Mapper.module
        else:
            Mapper.cuda()
            Mapper = torch.nn.parallel.DistributedDataParallel(Mapper, find_unused_parameters=True)
            Mapper_module = Mapper.module
    elif args.gpu is not None:
        Mapper = Mapper.cuda(args.gpu)
        Mapper_module = Mapper

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = 'module.' + key
                new_state_dict[key] = value
            Mapper.load_state_dict(new_state_dict)
            # optimizer.load_state_dict(checkpoint['optimizer'])

    optimizer = torch.optim.Adam(Mapper_module.parameters(), lr=args.lr)
    batch = args.batch_size
    attention_layer = args.attention_layer
    if args.latent_path:
        latent_code_init_load = torch.load(args.latent_path).cuda()

    if args.use_cluster:
        Mapper_module.store_clusters(initial_state)

    if args.rank == 0:
        pbar = tqdm(range(args.step))
        filetext = open(os.path.join(output_dir, 'video.txt'), 'w')
        lasttext = ''
        video_duration = 0.2
    else:
        pbar = range(args.step)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    for i in pbar:
        Mapper.train()
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        if t < 1.15:
            for name, param in Mapper_module.named_parameters():
                if name.startswith("attention") or name.startswith("initial"):
                    param.requires_grad = False
        else:
            for name, param in Mapper_module.named_parameters():
                if name.startswith("attention") or name.startswith("initial"):
                    param.requires_grad = True

        if args.latent_path:
            code_choose = torch.randint(len(latent_code_init_load), (batch, )).cuda()
            latent_code_init = latent_code_init_load[code_choose]
        else:
            # latent_code_init = mean_latent.detach().clone().repeat(batch, 18, 1)
            latent_code_init_not_trunc = torch.randn(batch, 512).cuda()
            with torch.no_grad():
                _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                            truncation=args.truncation, truncation_latent=mean_latent)

        if args.work_in_stylespace:
            with torch.no_grad():
                img_orig, _, latent_code_init, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True)
            latent = [s.detach().clone() for s in latent_code_init]
            # for c, s in enumerate(latent):
            #     if c in STYLESPACE_INDICES_WITHOUT_TORGB:
            #         s.requires_grad = True
        else:
            with torch.no_grad():
                img_orig, _, _, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True)
            latent = latent_code_init.detach().clone()

        with torch.no_grad():
            blend_feature = feature_map[attention_layer - 1]
            blend_size = blend_feature.shape[-1]
            feature_map.append(g_ema.input.input.repeat(batch, 1, 1, 1))

            phras_choose = torch.randint(len(phras), (batch, ))
            text1 = [phras[choose] for choose in phras_choose]
            phras_choose = torch.randint(len(phras), (batch*2, ))
            text2 = [phras[phras_choose[choose]] + " and " + phras[phras_choose[choose+batch]] for choose in range(batch)]

            # phras_choose = torch.randint(len(phras), (batch, ))
            # text1 = [phras[choose] for choose in phras_choose]
            # phras_choose = torch.randint(len(sentence), (batch, ))
            # text2 = [sentence[choose] for choose in phras_choose]

            phras_choose = torch.randn((batch, ))
            text = clip.tokenize([text1[i] if choose < 1 else text2[i] for (i, choose) in enumerate(phras_choose)], truncate=True).cuda()
            # phras_choose = (torch.ones((batch, )) * args.gpu).type(torch.int)
            # text = clip.tokenize([phras[choose] for choose in phras_choose], truncate=True).cuda()
            text_features_origin = clip_loss.model.encode_text(text)

            # text = torch.cat([clip.tokenize(args.description)] * batch).cuda()
            # text_features_origin = clip_loss.model.encode_text(text)

            text_source = torch.cat([clip.tokenize("Human face")] * batch).cuda()
            text_features_source = clip_loss.model.encode_text(text_source)

            attention_text = ["tanned skin", "narrow nose", "narrow eyes", "thin eyebrows", "wearing a pair of earrings", "pink lipsticks", "grey hair"]
            phras_choose = torch.randint(len(attention_text), (batch, ))
            # phras_choose = (torch.ones((batch, )) * args.gpu).type(torch.int)
            text_attention = clip.tokenize([attention_text[choose] for choose in phras_choose], truncate=True).cuda()
            # text_attention = torch.cat([clip.tokenize(args.attention_description)] * batch).cuda()
            text_features_attention = clip_loss.model.encode_text(text_attention).float()

            first_text_features_attention = text_features_attention[[0]].clone()
            if args.distributed:
                dist.broadcast(first_text_features_attention, 0)
            first_text_features_attention = first_text_features_attention.repeat(batch, 1)

            if not args.work_in_stylespace:
                _, attention_map, _ = Mapper(torch.cat([text_features_attention.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1), feature_map, blend_size, attention_text=first_text_features_attention)
            else:
                _, attention_map, _ = Mapper([torch.cat([text_features_attention.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in latent], feature_map, blend_size, attention_text=first_text_features_attention)

            img = img_orig.clone()
            # attention_map[attention_map < 0.4] = 0.4
            # img = (img + 1) * torch.nn.functional.interpolate(attention_map, (img.shape[2], img.shape[3])) - 1

            # boxes = masks_to_boxes(attention_map)
            # boxes[:, 0] = (boxes[:, 0] * img.shape[2] / attention_map.shape[2]).type(torch.int)
            # boxes[:, 1] = (boxes[:, 1] * img.shape[3] / attention_map.shape[3]).type(torch.int)
            # boxes[:, 2] = (boxes[:, 2] * img.shape[2] / attention_map.shape[2]).type(torch.int)
            # boxes[:, 3] = (boxes[:, 3] * img.shape[3] / attention_map.shape[3]).type(torch.int)
            # for k in range(attention_map.shape[0]):
            #     img[[k]] = torchvision.transforms.functional.resized_crop(img[[k]], boxes[k, 0], boxes[k, 1], boxes[k, 2] - boxes[k, 0], boxes[k, 3] - boxes[k, 1], (img.shape[2], img.shape[3]))
            
            image = clip_loss.avg_pool(clip_loss.upsample(img))
            image_features_origin = clip_loss.model.encode_image(image)

            epilsion = torch.nn.functional.normalize(torch.randn_like(image_features_origin), dim=-1)
            image_features_perturb = image_features_origin + 0.1 * epilsion * torch.norm(image_features_origin, dim=-1, keepdim=True).repeat(1, image_features_origin.shape[1])
            image_features_perturb = torch.nn.functional.normalize(image_features_perturb, dim=-1)

            clip_features_origin = torch.cat([image_features_perturb, text_features_origin], dim=0)
            shuffle = torch.randperm(clip_features_origin.shape[0]).to(clip_features_origin.device)
            clip_features_origin = image_features_origin

        # shuffle = torch.randperm(clip_features_origin.shape[0]).to(clip_features_origin.device)
        # clip_features_shuffle = clip_features_origin.clone()[shuffle]
        # delta_features_clip = clip_features_origin - clip_features_shuffle
        # diff_index = (torch.norm(delta_features_clip, dim=1) > 1e-4).to(clip_features_origin.device)
        # if sum(diff_index) < 1:
        #     shuffle = torch.randperm(clip_features_origin.shape[0]).to(clip_features_origin.device)
        #     clip_features_shuffle = clip_features_origin.clone()[shuffle]
        #     delta_features_clip = clip_features_origin - clip_features_shuffle
        #     diff_index = (torch.norm(delta_features_clip, dim=1) > 1e-4).to(clip_features_origin.device)

        if args.latent_path:
            code_choose = torch.randint(len(latent_code_init_load), (batch, )).cuda()
            latent_code_init = latent_code_init_load[code_choose]
        else:
            # latent_code_init = mean_latent.detach().clone().repeat(batch, 18, 1)
            latent_code_init_not_trunc = torch.randn(batch, 512).cuda()
            with torch.no_grad():
                _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                            truncation=args.truncation, truncation_latent=mean_latent)

        if args.work_in_stylespace:
            with torch.no_grad():
                img_orig, _, latent_code_init, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True)
            latent = [s.detach().clone() for s in latent_code_init]
            # for c, s in enumerate(latent):
            #     if c in STYLESPACE_INDICES_WITHOUT_TORGB:
            #         s.requires_grad = True
        else:
            with torch.no_grad():
                img_orig, _, _, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True)
            latent = latent_code_init.detach().clone()
        feature_map.append(g_ema.input.input.repeat(batch, 1, 1, 1))

        # Consisty Loss
        first_feature = []
        for j in range(len(feature_map)):
            first_feature_map = feature_map[j][[0]].clone()
            if args.distributed:
                dist.broadcast(first_feature_map, 0)
            first_feature.append(first_feature_map.repeat(batch, 1, 1, 1))
        first_blend_feature = first_feature[attention_layer - 1]
        if not args.work_in_stylespace:
            first_latent = latent[[0]].clone()
            if args.distributed:
                dist.broadcast(first_latent, 0)
            first_latent = first_latent.repeat(batch, 1, 1)
        else:
            first_latent = []
            for j in range(len(latent)):
                first_latent_code_init = latent[j][[0]].clone()
                if args.distributed:
                    dist.broadcast(first_latent_code_init, 0)
                first_latent.append(first_latent_code_init.repeat(batch, 1, 1, 1, 1))
        first_img_orig = img_orig[[0]].clone()
        if args.distributed:
            dist.broadcast(first_img_orig, 0)
        first_img_orig = first_img_orig.repeat(batch, 1, 1, 1)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if not args.work_in_stylespace:
                delta_zs, attention_map, delta_loss = Mapper(torch.cat([clip_features_origin.unsqueeze(1).repeat(1, first_latent.shape[1], 1), first_latent], dim=-1), first_feature, blend_size, attention_text=first_text_features_attention)
                ## image feature space
                strength = (1 + 0.2 * torch.rand((delta_zs.shape[0])).to(delta_zs.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs.shape[1:])
                strength = torch.ones_like(delta_zs)
                # strength[:, w_code_num[attention_layer]:, :] = 0.0
                new_latent_code = first_latent + strength * delta_zs
            else:
                new_latent_code, attention_map, delta_loss = Mapper([torch.cat([clip_features_origin.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in first_latent], first_feature, blend_size, attention_text=first_text_features_attention)
            loss_delta = delta_loss[0]
            loss_secphase = delta_loss[1]
            loss_essence = delta_loss[2]

            img_gen_shuffle, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace, attention_layer=attention_layer, attention_map=attention_map, feature_map=first_feature)
            
            img = img_gen_shuffle.clone()
            # attention_map[attention_map < 0.4] = 0.4
            # img = (img + 1) * torch.nn.functional.interpolate(attention_map, (img.shape[2], img.shape[3])) - 1

            # boxes = masks_to_boxes(attention_map)
            # boxes[:, 0] = (boxes[:, 0] * img.shape[2] / attention_map.shape[2]).type(torch.int)
            # boxes[:, 1] = (boxes[:, 1] * img.shape[3] / attention_map.shape[3]).type(torch.int)
            # boxes[:, 2] = (boxes[:, 2] * img.shape[2] / attention_map.shape[2]).type(torch.int)
            # boxes[:, 3] = (boxes[:, 3] * img.shape[3] / attention_map.shape[3]).type(torch.int)
            # for k in range(attention_map.shape[0]):
            #     img[[k]] = torchvision.transforms.functional.resized_crop(img[[k]], boxes[k, 0], boxes[k, 1], boxes[k, 2] - boxes[k, 0], boxes[k, 3] - boxes[k, 1], (img.shape[2], img.shape[3]))
            
            image_shuffle = clip_loss.avg_pool(clip_loss.upsample(img))
            image_features_shuffle = clip_loss.model.encode_image(image_shuffle)
            # loss_consist = torch.mean(1 - torch.nn.functional.cosine_similarity(image_features_shuffle, clip_features_shuffle))

            # if not args.work_in_stylespace:
            #     delta_zs, attention_map, delta_loss = Mapper(torch.cat([clip_features_origin.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1), feature_map, blend_size)
            #     # strength = (1 + 0.2 * torch.rand((delta_zs.shape[0])).to(delta_zs.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs.shape[1:])
            #     strength = torch.ones_like(delta_zs)
            #     # strength[:, w_code_num[attention_layer]:, :] = 0.0
            #     new_latent_code = latent + strength * delta_zs
            # else:
            #     new_latent_code, attention_map, delta_loss = Mapper([torch.cat([clip_features_origin.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in latent], feature_map, blend_size)
            # loss_delta = delta_loss[0]
            # loss_secphase = delta_loss[1]
            # loss_essence = delta_loss[2]
            # img_gen, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace, attention_layer=attention_layer, attention_map=attention_map, feature_map=feature_map)
            # image_gen = clip_loss.avg_pool(clip_loss.upsample(img_gen))
            # image_features_gen = clip_loss.model.encode_image(image_gen)
            loss_perceptual = perceptual_loss(img_gen_shuffle, first_img_orig)
            # target_point = clip_features_origin - 0.4 * text_features_source + 0.2 * image_features_origin

            # cutouts = normalize(make_cutouts(img_gen))
            # image_gen = clip_loss.avg_pool(torch.nn.Upsample(scale_factor=10)(img_gen))
            # cutouts = 2 * (make_cutouts(image_gen * 0.5 + 0.5)) - 1
            # image_features_gen = clip_loss.model.encode_image(cutouts)
            # target_point = torch.cat([target_point] * args.cutn, dim=0)

            # loss_consist = torch.nn.MSELoss()(image_features_gen, target_point)
            # loss_consist = torch.mean(1 - torch.nn.functional.cosine_similarity(image_features_gen, target_point))
            loss_identity = loss_perceptual

            # with torch.no_grad():
            #     loss_identity = id_loss(img_gen, img_orig)[0]
            
            # loss_consist = torch.mean(1 - torch.nn.functional.cosine_similarity(image_features_gen, clip_features_origin))
            # loss_secphase = torch.tensor([0.0]).to(loss_identity.device)
            # loss_essence = torch.tensor([0.0]).to(loss_identity.device)

            # strength = (1 + 0.2 * torch.rand((delta_zs.shape[0])).to(delta_zs.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs.shape[1:])
            # new_latent_code = latent - strength * delta_zs
            # img_gen_neg, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
            # image_gen_neg = clip_loss.avg_pool(clip_loss.upsample(img_gen_neg))
            # image_features_neg = clip_loss.model.encode_image(image_gen_neg)
            # loss_consist_neg = torch.mean(torch.nn.functional.cosine_similarity(image_features_neg, clip_features_origin))
            # loss_identity_neg = id_loss(img_gen_neg, img_orig)[0]

        ## 2nd-phase Loss
            # if sum(diff_index) < 1:
            #     loss_secphase = torch.tensor([0.0]).to(clip_features_origin.device)
            # else:
            #     delta_feature_img = image_features_gen - image_features_shuffle
            #     loss_secphase = 1 - torch.mean(torch.nn.functional.cosine_similarity(delta_feature_img[diff_index], delta_features_clip[diff_index]))

            if args.distributed:
                image_features_shuffle = torch.cat(GatherLayer.apply(image_features_shuffle), dim=0)
                clip_features_origin = torch.cat(GatherLayer.apply(clip_features_origin), dim=0)
            image_features_shuffle_norm = torch.nn.functional.normalize(image_features_shuffle, dim=-1)
            clip_features_shuffle_norm = torch.nn.functional.normalize(clip_features_origin, dim=-1)
            similiarity_map = image_features_shuffle_norm @ clip_features_shuffle_norm.T / 0.01
            loss_consist = torch.nn.functional.cross_entropy(similiarity_map, torch.arange(similiarity_map.shape[0]).to(similiarity_map.device))

            # delta_zs, delta_loss = Mapper(torch.cat([image_features_origin.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1))
            # loss_delta += delta_loss
            # strength = (1 + 0.2 * torch.rand((delta_zs.shape[0])).to(delta_zs.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs.shape[1:])
            # new_latent_code = latent + strength * delta_zs
            # img_gen_consist, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
            # image_consist = clip_loss.avg_pool(clip_loss.upsample(img_gen_consist))
            # image_features_consist = clip_loss.model.encode_image(image_consist)
            # loss_consist = torch.mean(1 - torch.nn.functional.cosine_similarity(image_features_consist, image_features_origin))
            # loss_identity = id_loss(img_gen_consist, img_orig)[0]

        ## Cycle Loss
        # phras_choose = torch.randint(len(phras), (batch, ))
        # text = clip.tokenize([phras[choose] for choose in phras_choose]).cuda()

        # # text = torch.cat([clip.tokenize(args.description)] * batch).cuda()

        # with torch.no_grad():
        #     text_features = clip_loss.model.encode_text(text)
        # with torch.cuda.amp.autocast(enabled=args.amp):
        #     delta_zs, delta_loss = Mapper(torch.cat([text_features.unsqueeze(1).repeat(1, latent.shape[1], 1), latent], dim=-1))
        #     loss_delta += delta_loss
        #     strength = (1 + 0.2 * torch.rand((delta_zs.shape[0])).to(delta_zs.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs.shape[1:])

        #     new_latent_code = latent + strength * delta_zs
        #     img_gen, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
        #     loss_cycle = torch.mean(torch.diagonal(clip_loss(img_gen, text)))
        #     loss_identity = id_loss(img_gen, img_orig)[0]

        #     new_latent_code = latent - strength * delta_zs
        #     img_gen, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)
        #     loss_cycle_neg = 1 - torch.mean(torch.diagonal(clip_loss(img_gen, text)))
        #     loss_identity_neg = id_loss(img_gen, img_orig)[0]

        ## 2nd-phase Loss
        # shuffle = torch.randperm(clip_features_origin.shape[0]).to(clip_features_origin.device)
        # clip_feature_shuffle = clip_features_origin.clone()[shuffle]
        # delta_feature_clip = clip_features_origin - clip_feature_shuffle
        # diff_index = (torch.norm(delta_feature_clip, dim=1) > 1e-4).to(clip_features_origin.device)
        # if sum(diff_index) < 1:
        #     shuffle = torch.randperm(clip_features_origin.shape[0]).to(clip_features_origin.device)
        #     clip_feature_shuffle = clip_features_origin.clone()[shuffle]
        #     delta_feature_clip = clip_features_origin - clip_feature_shuffle
        #     diff_index = (torch.norm(delta_feature_clip, dim=1) > 1e-4).to(clip_features_origin.device)
        # if sum(diff_index) < 1:
        #     loss_secphase = torch.tensor([0.0]).to(clip_features_origin.device)
        # else:
        #     clip_feature_shuffle = clip_feature_shuffle[diff_index]
        #     delta_feature_clip = delta_feature_clip[diff_index]
        #     with torch.cuda.amp.autocast(enabled=args.amp):
        #         delta_zs_shuffle, delta_loss = Mapper(torch.cat([clip_feature_shuffle.unsqueeze(1).repeat(1, latent.shape[1], 1), latent[diff_index]], dim=-1))
        #         loss_delta += delta_loss
        #         strength = (1 + 0.2 * torch.rand((delta_zs_shuffle.shape[0])).to(delta_zs_shuffle.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs_shuffle.shape[1:])
        #         new_latent_code_shuffle = latent[diff_index] + strength * delta_zs_shuffle
        #         img_gen_shuffle, _ = g_ema([new_latent_code_shuffle], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

        #         image_gen = clip_loss.avg_pool(clip_loss.upsample(img_gen[diff_index]))
        #         image_features = clip_loss.model.encode_image(image_gen)
        #         image_shuffle = clip_loss.avg_pool(clip_loss.upsample(img_gen_shuffle))
        #         image_features_shuffle = clip_loss.model.encode_image(image_shuffle)
        #         delta_feature_img = image_features - image_features_shuffle
        #         loss_secphase = 1 - torch.mean(torch.nn.functional.cosine_similarity(delta_feature_img, delta_feature_clip))

        ## Essence Loss
        # first_text_feature = clip_features_origin[0, :].clone()
        # if args.distributed:
        #     dist.broadcast(first_text_feature, 0)
        with torch.cuda.amp.autocast(enabled=args.amp):
        #     delta_zs, delta_loss = Mapper(torch.cat([first_text_feature.unsqueeze(0).unsqueeze(0).repeat(batch, latent.shape[1], 1), latent], dim=-1))
        #     loss_delta += delta_loss
        #     strength = (1 + 0.2 * torch.rand((delta_zs.shape[0])).to(delta_zs.device)).unsqueeze(1).unsqueeze(1).repeat(1, *delta_zs.shape[1:])
        #     new_latent_code = latent + strength * delta_zs

        #     img_gen_essence, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

        #     image_essence = clip_loss.avg_pool(clip_loss.upsample(img_gen_essence))
        #     image_features_essence = clip_loss.model.encode_image(image_essence)
        #     delta_feature = image_features_essence - image_features_origin
        #     delta_feature = delta_feature / delta_feature.norm(dim=-1, keepdim=True)
        #     if args.distributed:
        #         delta_feature = torch.cat(GatherLayer.apply(delta_feature), dim=0)
        #     diffs_mat_amp = delta_feature @ delta_feature.T
        #     ones_mat = torch.ones(diffs_mat_amp.shape[0]).cuda()
        #     loss_essence = torch.sum(ones_mat - diffs_mat_amp) / (diffs_mat_amp.shape[0] ** 2 - diffs_mat_amp.shape[0])
            # loss_essence = -torch.mean(torch.nn.functional.cosine_similarity(delta_feature[1:], delta_feature[:-1]))

            # loss_identity = id_loss(img_gen_essence, img_orig)[0]

            # img_orig_idfeats = id_loss.extract_feats(img_orig)
            # img_gen_essence_idfeats = id_loss.extract_feats(img_gen_essence)
            # if args.distributed:
            #     img_orig_idfeats = torch.cat(GatherLayer.apply(img_orig_idfeats), dim=0)
            #     img_gen_essence_idfeats = torch.cat(GatherLayer.apply(img_gen_essence_idfeats), dim=0)
            # similiarity_map = img_orig_idfeats @ img_gen_essence_idfeats.T / 0.01
            # loss_identity = torch.nn.functional.cross_entropy(similiarity_map, torch.arange(similiarity_map.shape[0]).to(similiarity_map.device))

            loss_total = loss_consist + max(0, min(1, (t - 0.15) / 0.1)) * (args.lambda_ess * loss_essence + args.lambda_sec * loss_secphase) + max(0, min(1, (t - 0.05) / 0.1)) * (args.lambda_id * loss_identity) + args.lambda_delta * loss_delta

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            optimizer.step()
        
        if args.rank == 0:
            pbar.set_description(
                (
                    f"loss: {loss_consist.item():.4f}; {loss_essence.item():.4f}; {loss_secphase.item():.4f}; {loss_identity.item():.4f}; {loss_delta.item():.4f}"
                )
            )
            writer.add_scalar('loss/total', loss_total.item(), i)
            writer.add_scalar('loss/consist', loss_consist.item(), i)
            writer.add_scalar('loss/essence', loss_identity.item(), i)
            writer.add_scalar('loss/cycle', loss_secphase.item(), i)
            if args.save_intermediate_image_every > 0 and (i + 1) % args.save_intermediate_image_every == 0:
                torch.save(Mapper.state_dict(), os.path.join(output_dir, f"{str(i+1).zfill(5)}_mapper.pt"))
                Mapper.eval()
                imgs = []
                attentions = []
                with torch.no_grad():
                    for k in range(len(phras_own)):
                        text_inputs = torch.cat([clip.tokenize(phras_own[k])]).cuda()
                        text_features = clip_loss.model.encode_text(text_inputs)
                        # text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
                        # text_features = clip_loss.model.encode_text(text_inputs)
                        first_feature_map = []
                        for j in range(len(feature_map)):
                            first_feature_map.append(feature_map[j][[0]])
                        blend_feature = first_feature_map[attention_layer - 1]
                        blend_size = blend_feature.shape[-1]
                        if not args.work_in_stylespace:
                            first_latent_code_init = latent[0].unsqueeze(0)
                            delta_zs, attention_map, _ = Mapper_module.forward(torch.cat([text_features.unsqueeze(1).repeat(1, latent.shape[1], 1), first_latent_code_init], dim=-1), first_feature_map, blend_size)
                            strength = torch.ones_like(delta_zs)
                            # strength[:, w_code_num[attention_layer]:, :] = 0.0
                            new_latent_code = first_latent_code_init + strength * delta_zs
                        else:
                            first_latent_code_init = []
                            for j in range(len(latent)):
                                first_latent_code_init.append(latent[j][0].unsqueeze(0))
                            new_latent_code, attention_map, _ = Mapper_module.forward([torch.cat([text_features.unsqueeze(1), s[:, :, :, 0, 0]], dim=-1) for s in first_latent_code_init], first_feature_map, blend_size)
                    
                        # attention_map[attention_map<0.8] = 0.0
                        img_gen, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace, attention_layer=attention_layer, attention_map=attention_map, feature_map=first_feature_map)
                        imgs.append(img_gen)
                        attentions.append(attention_map)
                imgs = torch.cat(imgs)
                attentions = torch.cat(attentions)
                torchvision.utils.save_image(imgs.detach().cpu(), os.path.join(output_dir, f"{str(i+1).zfill(5)}.jpg"), nrow=img_orig.shape[0], normalize=True, range=(-1, 1))
                torchvision.utils.save_image(attentions.detach().cpu(), os.path.join(output_dir, f"attention{str(i+1).zfill(5)}.jpg"), nrow=img_orig.shape[0], normalize=True, range=(0, 1))
                filetext.write(f"file ./{str(i+1).zfill(5)}.jpg\n")
                filetext.write(f"duration {str(video_duration)}\n")
                lasttext = f"file ./{str(i+1).zfill(5)}.jpg"
                # IS, FID, ID = cal_evaluation(args, attention_layer, output_dir, phras, g_ema, Mapper_module, clip_loss.model, id_loss, iteration=100, batch=2, dataset_dir='../data/CelebAMask-HQ/CelebA-HQ-img')
                # writer.add_scalar('metric/is', IS, i)
                # writer.add_scalar('metric/fid', FID, i)
                # writer.add_scalar('metric/id', ID, i)
                del first_feature_map, first_latent_code_init
        # torch.cuda.empty_cache()
    if args.rank == 0:
        filetext.write(lasttext)
        filetext.close()

        print(f"loss: {loss_consist.item():.4f}; {loss_essence.item():.4f}; {loss_secphase.item():.4f}; {loss_identity.item():.4f}; {loss_delta.item():.4f}")
        torch.save(Mapper.state_dict(), os.path.join(output_dir, "final_mapper.pt"))
        save_batch = min(4, 2 * batch)
        text_attention = torch.cat([clip.tokenize(args.attention_description)] * save_batch).cuda()
        text_features_attention = clip_loss.model.encode_text(text_attention).float()
        Mapper.eval()

        if args.latent_path:
            code_choose = torch.randint(len(latent_code_init_load), (save_batch, )).cuda()
            latent_code_init = latent_code_init_load[code_choose]
        else:
            latent_code_init_not_trunc = torch.randn(save_batch, 512).cuda()
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
            feature_map.append(g_ema.input.input.repeat(save_batch, 1, 1, 1))

        blend_feature = feature_map[attention_layer - 1]
        blend_size = blend_feature.shape[-1]
        final_result = [img_orig.detach().cpu()]
        attention_maps = []
        for j in range(len(phras_own)):
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(phras_own[j])]).cuda()
                text_features = clip_loss.model.encode_text(text_inputs)
                if not args.work_in_stylespace:
                    delta_zs, attention_map, _ = Mapper_module.forward(torch.cat([text_features.unsqueeze(1).repeat(save_batch, latent_code_init.shape[1], 1), latent_code_init], dim=-1), feature_map, blend_size)
                    strength = torch.ones_like(delta_zs)
                    # strength[:, w_code_num[attention_layer]:, :] = 0.0
                    new_latent_code = latent_code_init + strength * delta_zs
                else:
                    new_latent_code, attention_map, _ = Mapper_module.forward([torch.cat([text_features.unsqueeze(1).repeat(save_batch, 1, 1), s[:, :, :, 0, 0]], dim=-1) for s in latent], feature_map, blend_size)
                
                # attention_map[attention_map<0.8] = 0.0
                img_gen, _ = g_ema([new_latent_code], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace, attention_layer=attention_layer, attention_map=attention_map, feature_map=feature_map)
                final_result.append(img_gen.detach().cpu())
                attention_maps.append(attention_map.detach().cpu())

        final_result = torch.cat(final_result)
        attention_maps = torch.cat(attention_maps)
        torchvision.utils.save_image(final_result, os.path.join(output_dir, "final_result.jpg"), 
            nrow=img_orig.shape[0], normalize=True, scale_each=True, range=(-1, 1))
        torchvision.utils.save_image(attention_maps, os.path.join(output_dir, "final_attention.jpg"), nrow=img_orig.shape[0], normalize=True, scale_each=True, range=(0, 1))
        output_dirs = [output_dir]
        # mean_IOU = calculate_IOU(args, attention_layer, blend_size, g_ema, Mapper_module, clip_loss.model)
        # print(mean_IOU)
    else:
        output_dirs = [None]
    # dist.broadcast_object_list(output_dirs, src=0)
    # IS, FID, ID, improve = cal_evaluation(args, attention_layer, output_dirs[0], phras, g_ema, Mapper_module, clip_loss.model, id_loss, iteration=5000//dist.get_world_size(), batch=2, dataset_dir='../data/CelebAMask-HQ/CelebA-HQ-img', one_gpu=False)
    # print(IS, FID, ID, improve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description_dir", type=str, default="../celeba-caption", help="the corpus of face descriptions")
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the editing/generation")
    parser.add_argument("--attention_description", type=str, default="blonde hair", help="the text that guides where to edit/generate")
    parser.add_argument("--own_description_dir", type=str, default="./my_phras_simple.txt", help="the corpus of specific descriptions")
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel_multiplier")
    parser.add_argument("--attention_layer", type=int, default=8, help="blende attention layer")
    parser.add_argument('--use_cluster', default=False, action='store_true')
    parser.add_argument("--cluster_path", type=str, default=None, help="k-means cluster"
                                                                      "Expects a .pt format")
    parser.add_argument("--cluster_layer", type=int, default=13, help="cluster layer")
    parser.add_argument("--cluster_num", type=int, default=10, help="cluster num")
    parser.add_argument("--batch_size", type=int, default=1, help="traning batchsize")
    parser.add_argument("--cutn", type=int, default=64, help="cutout nums")
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
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest mapper checkpoint (default: none)')

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