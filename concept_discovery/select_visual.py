import sklearn
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency

from tqdm import tqdm
import os
import numpy as np
import torch
import open_clip
from PIL import Image
from torch import nn
import pandas as pd
import json
import sys
import argparse


device = 'cuda:0'


def cal_mi(concept_image_dict_index,image_embeddings,concept_embeddings):

    with torch.no_grad():
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        concept_embeddings /= concept_embeddings.norm(dim=-1, keepdim=True)
        mutual_info = []
        for i in tqdm(range(concept_embeddings.shape[0])):
            x = []
            pos_ids = concept_image_dict_index[i]
            neg_ids = [i for i in range(concept_embeddings.shape[0]) if i not in pos_ids]
            pos_ct= min(len(pos_ids),10)
            neg_ct= 10
            pos_ids = np.random.sample(pos_ids,pos_ct,replace = False)
            neg_ids = np.random.sample(pos_ids,neg_ct,replace = False)
            choose_id = np.concatenate(pos_ids,neg_ids,axis=0)
            x = concept_embeddings[i]@image_embeddings[choose_id].T
            x = x.squeeze(0)
            x = x.cpu().numpy()
            y = [1 for _ in range(pos_ct)] + [0 for _ in range(neg_ct)]
            mutual_info.append(mutual_info_regression(y, x)[0])

    mutual_info = np.array(mutual_info)

    return mutual_info

def main():
    with open('concept_image_dict_index.json','r') as fp:
        concept_image_dict_index = json.load(fp)
    with open('concepts.json','r') as fp:
        concepts = json.load(fp)
    image_embeddings = torch.load('image_embeddings.pt')
    concept_embeddings = torch.load('concept_embeddings.pt')
    mi = cal_mi(concept_image_dict_index,image_embeddings,concept_embeddings)
    rank = np.argsort(mi)[::-1]
    #Select top k visually discriminative concepts, the k can be decided according to the scale of your downstream tasks
    k = 100000
    selected_concepts = []
    for i in range(k):
        selected_concepts.append(concepts[rank[i]])

    with open('visual_concepts.json','w') as fw:
        json.dump(selected_concepts,fw)
if __name__ == "__main__":
    main()
