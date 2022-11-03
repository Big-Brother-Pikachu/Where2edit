from email.policy import default
import sys
sys.path.append("..")

import os
import argparse
import re
import numpy as np
from PIL import Image
import streamlit as st
import torch
import clip
import random
import torchvision

from attention.attention_model import Generator, StyledConv, ModulatedConv2d, PixelNorm, EqualLinear, EqualConv2d
from utils import ensure_checkpoint_exists, descripition_corpus, set_random_seed, pairwise_distance, CA_NET

from utils_demo import FullSpaceMapperSpatialLin_Net, one_text_edit, transform_img, load_e4e_standalone


@st.cache(suppress_st_warning=True)
def load_models():
    ensure_checkpoint_exists("../pretrained_models/stylegan2-ffhq-config-f-1024.pt")

    image_size = 1024
    attention_layer = 13
    channel_multiplier = 2
    g_ema = Generator(image_size, 512, 8, channel_multiplier=channel_multiplier)
    g_ema.load_state_dict(torch.load("../pretrained_models/stylegan2-ffhq-config-f-1024.pt")["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)
    model, _ = clip.load("ViT-B/32", device="cuda", download_root='../pretrained_models')

    Mapper = FullSpaceMapperSpatialLin_Net(g_ema.n_latent, model.visual.output_dim + mean_latent.shape[-1], mean_latent.shape[-1], attention_layer=attention_layer, channel_multiplier=channel_multiplier,  clusters=20, cluster_dim=576, cluster_layer=13)
    state_dict = torch.load('./models/final_mapper.pt')
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:]
        new_state_dict[new_key] = value
    Mapper.load_state_dict(new_state_dict, strict=True)
    Mapper.eval()
    Mapper = Mapper.cuda()

    ckpt = torch.load('../pretrained_models/e4e_ffhq_encode.pt', map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = '../pretrained_models/e4e_ffhq_encode.pt'
    opts = argparse.Namespace(**opts)
    net_e4e = load_e4e_standalone('../pretrained_models/e4e_ffhq_encode.pt')
    net_e4e.eval()
    net_e4e = net_e4e.cuda()
    return g_ema, model, Mapper, net_e4e


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    st.title("Text-Guided Editing of Faces")

    mode = st.sidebar.selectbox('Image Mode', ('Real', 'Syn'), index=0)

    if mode == 'Real':
        select = st.sidebar.selectbox('Use...', ('Provided Images', 'Your Own Image'), index=0)
        if select == 'Provided Images':
            choose = st.sidebar.selectbox("Which is...", ('Taylor Swift', 'Elon Musk', 'Angela Merkel', 'Leonardo DiCaprio', 'Portrait'))
            uploaded_file = './imgs/' + re.split((' '), choose)[-1] + '.png'
        else:
            uploaded_file = st.sidebar.file_uploader("Choose your own image...", type=['png', 'jpg'])

    description = st.sidebar.text_input('Description', 'Purple Hair')
    attention = st.sidebar.selectbox('Attention Description', ('', 'Skin', 'Nose', 'Eye', 'Eyebrow', 'Ear', 'Mouth', 'Hair'))
    if mode == 'Syn':
        seed = st.sidebar.number_input('Seed', value=150)
    strength_alpha = st.sidebar.slider('Editing Strength', min_value=0., max_value=0.3, value=0.1, step=0.01)
    attention_threshold = st.sidebar.slider('Attention Coverage', min_value=0., max_value=1., value=0.8, step=0.1)
    attention_threshold = 1.0 - 0.25 * attention_threshold

    image_size = 1024
    work_in_stylespace = True
    attention_layer = 13

    if mode == 'Syn':
        set_random_seed(seed)
    g_ema, model, Mapper, net_e4e = load_models()
    if mode == 'Syn':
        set_random_seed(seed)
    
    mean_latent = g_ema.mean_latent(4096)

    upsample = torch.nn.Upsample(scale_factor=7)
    avg_pool = torch.nn.AvgPool2d(kernel_size=image_size // 32)

    if mode == 'Real':
        if uploaded_file is None:
            return
        image = Image.open(uploaded_file)
        image = Image.Image.convert(image, 'RGB')
        img = transform_img(256, False, True, True, True)(image).unsqueeze(0)
        latents =  net_e4e(img.cuda())
        _, latents, styles = g_ema([latents], input_is_latent=True, return_latents=True)
        if work_in_stylespace:
            latent_code_init = styles
        else:
            latent_code_init = latents
    else:
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=0.7, truncation_latent=mean_latent)

        if work_in_stylespace:
            with torch.no_grad():
                _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
                latent_code_init = [s.detach().clone() for s in latent_code_init]
        else:
            latent_code_init = latent_code_init.detach().clone()

    col1, col2, col3 = st.columns(3)

    sequence_text = [description]
    with torch.no_grad():
        img_orig, _, _, feature_map = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False, return_features=True, input_is_stylespace=work_in_stylespace)
        image_orig = avg_pool(upsample(img_orig))
        image_features_origin = model.encode_image(image_orig)
        feature_map.append(g_ema.input.input)

        with col1:
            st.header("Original")
            st.image(0.5 * (img_orig + 1).detach().permute(0, 2, 3, 1).cpu().numpy(), use_column_width=True, clamp=True)

        final_attention = []
        if attention == 'Skin':
            attention_description = 'Tanned Skin'
        elif attention == 'Nose':
            attention_description = 'Narrow Nose'
        elif attention == 'Eye':
            attention_description = 'Narrow Eye'
        elif attention == 'Eyebrow':
            attention_description = 'Thin Eyebrows'
        elif attention == 'Ear':
            attention_description = 'Wearing a pair of Earrings'
        elif attention == 'Mouth':
            attention_description = 'Pink Lipsticks'
        elif attention == 'Hair':
            attention_description = 'Grey Hair'
        for text in sequence_text:
            text_inputs = clip.tokenize(text).cuda()
            text_features = model.encode_text(text_inputs)
            if attention == '':
                attention_description = text
            attention_text_inputs = clip.tokenize(attention_description).cuda()
            attention_text_features = model.encode_text(attention_text_inputs).float()
            img_gen, new_latent_code, attention_map, _ = one_text_edit(text_features, attention_text_features, latent_code_init, g_ema, Mapper, attention_layer, work_in_stylespace, strength_alpha, attention_threshold, feature_map)
            image_gen = avg_pool(upsample(img_gen))
            image_features_gen = model.encode_image(image_gen)

            sim = torch.cosine_similarity(image_features_origin, text_features[:, :])
            print(sim)
            sim = torch.cosine_similarity(image_features_gen, text_features[:, :])
            print(sim)
            final_attention.append(attention_map)

    final_attention = torch.cat(final_attention)
    print(description, attention_description)

    with col2:
        st.header('Edited')
        st.image(0.5 * (img_gen + 1).detach().permute(0, 2, 3, 1).cpu().numpy(), use_column_width=True, clamp=True, caption='+' + description)

    with col3:
        st.header('Attention')
        st.image(attention_map.detach().permute(0, 2, 3, 1).cpu().numpy(), use_column_width=True, clamp=True, caption='Prompt: ' + attention_description)

if __name__ == '__main__':
  main()
