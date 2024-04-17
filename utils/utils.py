import torch
import numpy as np
from tqdm import tqdm
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