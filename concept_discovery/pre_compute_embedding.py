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

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def process_text(texts,tokenizer,model_text):
    device = 'cuda:0'
    with torch.no_grad():
        texts = tokenizer(texts)
        texts = texts.to(device)
        text_feature = model_text(texts)
    return text_feature

def process_image(imgs,model_image):
    device = 'cuda:0'
    with torch.no_grad():
        imgs = imgs.to(device)
        image_feature = model_image(imgs)
    return image_feature

def batchify_run_text(process_fn,tokenizer,model_text, data_lst, res, batch_size, use_tqdm=True):
    data_lst_len = len(data_lst)
    num_batch = np.ceil(data_lst_len / batch_size).astype(int)
    iterator = range(num_batch)
    if use_tqdm:
        iterator = tqdm(iterator)
    for i in iterator:
        batch_data = data_lst[i * batch_size:(i + 1) * batch_size]
        batch_res = process_fn(batch_data,tokenizer,model_text)
        res[i * batch_size:(i + 1) * batch_size] = batch_res
        del batch_res
def batchify_run_image(process_fn,model_image, data_lst, res, batch_size, use_tqdm=True):
    data_lst_len = len(data_lst)
    num_batch = np.ceil(data_lst_len / batch_size).astype(int)
    iterator = range(num_batch)
    if use_tqdm:
        iterator = tqdm(iterator)
    for i in iterator:
        batch_data = data_lst[i * batch_size:(i + 1) * batch_size]
        batch_res = process_fn(batch_data,model_image)
        res[i * batch_size:(i + 1) * batch_size] = batch_res
        del batch_res
def main():
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    model.to(device)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text)
    model_image = torch.nn.DataParallel(model_image)
    with open('new_concepts.json', 'r') as fp:
        concepts = json.load(fp)
    with open('new_concept_image_dict.json', 'r') as fp:
        concept_image_dict = json.load(fp)
    images = []
    for path in os.listdir('images'):
        images.append(test_preprocess(Image.open(path)).unsqueeze(0))
    image_embeddings = torch.empty((len(images),768))
    concept_embeddings = torch.empty((len(concepts), 768))
    batchify_run_image(process_image,model_image,images,image_embeddings,1024)
    batchify_run_text(process_text,tokenizer,model_text,concepts,concept_embeddings,1024)
    concept_image_dict_index = {}
    for i in range(len(concepts)):
        concept_image_dict_index[i] = concept_image_dict[concepts[i]]
    torch.save(image_embeddings,'image_embeddings.pt')
    torch.save(concept_embeddings,'concept_embeddings.pt')
    with open('concept_image_dict_index.json','w') as fw:
        json.dump(concept_image_dict_index,fw)
if __name__ == "__main__":
    main()



