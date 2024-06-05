import sys

import numpy as np
import torch
import clip

from PIL import Image
import os
import argparse
import json
from tqdm import tqdm
import torch.nn as nn
import open_clip
data_root = 'data/'
shots = ['all']
num_concept = int(sys.argv[1])
device = 'cuda:0'
from utils.models import TextCLIP
from utils.utils import process_text,batchify_run_text


def main(args):
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model.load_state_dict(torch.load(os.path.join(args.dataset, 'checkpoint', 'epoch_' + args.epoch + '.pt')))
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model_text = TextCLIP(model)
    model_text.eval()
    model_text.to(device)
    with open(os.path.join('..',args.dataset,'category.json'),'r') as fp:
        categories = json.load(fp)
    with open(os.path.join('..',args.dataset,'all_concepts.json'),'r') as fp:
        texts = json.load(fp)
    with torch.no_grad():

        cls_truth = np.load(os.path.join('..',args.dataset,'class_label.npy'))
        cls_truth = torch.Tensor(cls_truth).t()
        cls_truth = cls_truth.to(device)
        cls_truth /= cls_truth.sum(dim=0)
        train_y = torch.load(os.path.join('..',args.dataset,'data','train_data_allshot.pt'))['label']


        image_embeddings = torch.load(os.path.join('..',args.dataset, 'img_embedding', 'train_allshot.pt'))

        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = torch.empty((len(texts),768))
        batchify_run_text(process_text,tokenizer,model_text,texts,text_embeddings,1024)
        predicts = image_embeddings @ text_embeddings.T
        predicts = predicts @ cls_truth
        predict_labels = torch.argmax(predicts, dim=1)
        pred_labels = predict_labels.cpu().numpy()
        correct = torch.sum(predict_labels == train_y)
        print('acc:',correct / train_y.shape[0])
        np.save(os.path.join('..',args.dataset,'pred_labels.npy'), pred_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    main(args)