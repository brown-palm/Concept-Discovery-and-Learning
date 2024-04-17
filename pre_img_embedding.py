import argparse
import json
import os
import logging

import pandas as pd
import numpy as np
import torch
import open_clip
from PIL import Image


from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

from torch import nn

from utils.models import TextCLIP,ImageCLIP

def main(args):
    if not os.path.exists(os.path.join(args.dataset,'img_embedding')):
        os.mkdir(os.path.join(args.dataset,'img_embedding'))
    if not os.path.exists(os.path.join(args.dataset,'txt_embedding')):
        os.mkdir(os.path.join(args.dataset,'txt_embedding'))

    if args.dataset == 'Flower':
        shots = ['1','2','4','8','all']
    else:
        shots = ['1', '2', '4', '8', '16', 'all']

    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    if args.learn:
        model.load_state_dict(torch.load(os.path.join(args.dataset,'checkpoint','epoch_'+args.epoch+'.pt')))
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model.to(device)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text)
    model_image = torch.nn.DataParallel(model_image)

    def process_text(texts):
        device = 'cuda:0'
        with torch.no_grad():
            texts = tokenizer(texts)
            texts = texts.to(device)
            text_feature = model_text(texts)
        return text_feature

    def process_image(imgs):
        device = 'cuda:0'
        with torch.no_grad():
            imgs = imgs.to(device)
            image_feature = model_image(imgs)
        return image_feature

    def batchify_run(process_fn, data_lst, res, batch_size, use_tqdm=True):
        data_lst_len = len(data_lst)
        num_batch = np.ceil(data_lst_len / batch_size).astype(int)
        iterator = range(num_batch)
        if use_tqdm:
            iterator = tqdm(iterator)
        for i in iterator:
            batch_data = data_lst[i * batch_size:(i + 1) * batch_size]
            batch_res = process_fn(batch_data)
            res[i * batch_size:(i + 1) * batch_size] = batch_res
            del batch_res
    if os.path.exists(os.path.join(args.dataset,'all_concepts.json')):

        with open(os.path.join(args.dataset,'all_concepts.json'),'r') as fp:
            all_concepts=json.load(fp)
        if args.learn:
            path = os.path.join(args.dataset,'txt_embedding_t', 'all.pt')
        else:
            path = os.path.join(args.dataset, 'txt_embedding', 'all.pt')
        if not os.path.exists(path):
            all_text_features = torch.empty((len(all_concepts), 768))
            batchify_run(process_text, all_concepts, all_text_features, 1024)
            torch.save(all_text_features,path)

    for shot in shots:

        train_images = torch.load(os.path.join(args.dataset,'data', 'train_data_'+shot+'shot.pt'))['image']
        if args.learn:
            path = os.path.join(args.dataset,'img_embedding_t','train_'+shot+'shot.pt')
        else:
            path = os.path.join(args.dataset, 'img_embedding', 'train_' + shot + 'shot.pt')
        if os.path.exists(path):
            pass
        else:
            train_image_features = torch.empty((len(train_images),768))
            batchify_run(process_image, train_images, train_image_features, 1024)

            torch.save(train_image_features,path)
    test_images = torch.load(os.path.join(args.dataset,'data', 'test_data.pt'))['image']
    if args.learn:
        path = os.path.join(args.dataset,'img_embedding_t','test.pt')
    else:
        path = os.path.join(args.dataset, 'img_embedding', 'test.pt')
    if os.path.exists(path):
        pass
    else:
        test_image_features = torch.empty((len(test_images), 768))
        batchify_run(process_image, test_images, test_image_features, 1024)
        torch.save(test_image_features,path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('learn', type=str)
    parser.add_argument('epoch', type=str)
    args = parser.parse_args()
    main(args)
