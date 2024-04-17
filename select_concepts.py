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


def cal_mi(dataset,shot,learn,alpha):
    with torch.no_grad():

        data = torch.load(os.path.join(dataset,'data','train_data_'+shot+'shot.pt'))
        images = data['image']
        labels = data['label']
        with open(os.path.join(dataset,'all_concepts.json'),'r') as fp:
            concepts=json.load(fp)
        class_label=np.load(os.path.join(dataset,'class_label.npy'))
        class_label_t = class_label.T
        cats = []
        for i in range(len(concepts)):
            cats.append(np.sum(class_label_t[i])/class_label_t.shape[1])
        if learn:
            image_features = torch.load(os.path.join(dataset,'img_embedding_t/train_'+shot+'shot.pt'))
            text_features = torch.load(os.path.join(dataset,'txt_embedding_t/'+'all.pt'))
        else:
            image_features = torch.load(os.path.join(dataset, 'img_embedding/train_' + shot + 'shot.pt'))
            text_features = torch.load(os.path.join(dataset, 'txt_embedding/' + 'all.pt'))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        mutual_info = []
        for i in tqdm(range(len(concepts))):
            x = []
            cand_ids = np.random.choice(len(labels),len(labels),replace=False)
            pos_ct=0
            neg_ct=0
            num_samples_pos = 30
            if shot == '1':
                num_samples_neg = 5
            if shot == '2':
                num_samples_neg = 10
            if shot == '4':
                num_samples_neg = 15
            if shot == '8':
                num_samples_neg = 20
            if shot == '16':
                num_samples_neg = 30
            if shot == 'all':
                num_samples_neg = 30
            choose_id = []
            #print(cand_ids)
            #print(labels)
            for c_id in cand_ids:
                if class_label[labels[c_id]][i]==1:
                    if pos_ct>=num_samples_pos:
                        continue
                    choose_id.append(c_id)
                    x.append([1])
                    pos_ct+=1
            for c_id in cand_ids:
                if class_label[labels[c_id]][i] == 0:
                    if neg_ct >= num_samples_neg:
                        continue
                    choose_id.append(c_id)
                    x.append([0])
                    neg_ct += 1
            choose_id = np.array(choose_id)
            y = text_features[i]@image_features[choose_id].T
            y = y.squeeze(0)
            y = y.cpu().numpy()
            x = np.array(x)

            mutual_info.append(alpha*mutual_info_regression(x, y)[0]+(1-alpha)*cats[i])

    #ids = np.argsort(mutual_info)[::-1]
    mutual_info = np.array(mutual_info)
    #print(0,mutual_info[ids[0]])
    #print(100,mutual_info[ids[100]])
    #print(500,mutual_info[ids[500]])
    #np.save(os.path.join(dataset,'mutual_info_'+shot+'shot.npy'),mutual_info)
    return mutual_info
def select_concepts(mutual_info, dataset,shot,num_classes,num_concept,learn):
    #mutual_info = np.load(os.path.join(dataset,'mutual_info_'+shot+'shot.npy'))
    ids = np.argsort(mutual_info)[::-1]
    with open(os.path.join(dataset,'all_concepts.json'), 'r') as fp:
        concepts = json.load(fp)

    #print(concepts[176])
    class_label=np.load(os.path.join(dataset,'class_label.npy'))
    class_label = class_label.T

    class_ct = {}
    class_concept_id = {}
    for c in range(num_classes):
        class_ct[c]=0
        class_concept_id[c]=[]
    new_concepts_id=[]
    for i in range(len(class_label)):
        c_id = ids[i]
        flag = 0
        for c in class_ct:
            if class_ct[c] < 1:
                flag = 1
        if flag == 0:
            print('f',i)
            break


        for c in range(num_classes):
            if class_label[c_id][c]==1:
                if class_ct[c]>=1:
                    continue
                else:
                    class_ct[c] += 1
                    if c_id in new_concepts_id:
                        continue
                    new_concepts_id.append(c_id)
                    for c in range(num_classes):
                        if class_label[c_id][c] == 1:
                            class_concept_id[c].append(c_id)

    for i in range(len(ids)):
        c_id = ids[i]
        if len(new_concepts_id)>=num_concept:
            break

        for c in range(num_classes):
            if class_label[c_id][c]==1:

                class_ct[c] += 1
                if c_id in new_concepts_id:
                    continue
                new_concepts_id.append(c_id)
                for c in range(num_classes):
                    if class_label[c_id][c] == 1:
                        class_concept_id[c].append(c_id)
                break
    new_concepts=[]
    new_class_label=class_label[new_concepts_id]
    for i in new_concepts_id:
        new_concepts.append(concepts[i])

    new_class_label = new_class_label.T
    if learn:
        with open(os.path.join(dataset,'select_concept_'+shot+'shot'+str(num_concept)+'_t.json'),'w') as fw:
            json.dump(new_concepts,fw)
        np.save(os.path.join(dataset,'class_label_'+shot+'shot'+str(num_concept)+'_t.npy'),new_class_label)
    else:
        with open(os.path.join(dataset,'select_concept_'+shot+'shot'+str(num_concept)+'.json'),'w') as fw:
            json.dump(new_concepts,fw)
        np.save(os.path.join(dataset,'class_label_'+shot+'shot'+str(num_concept)+'.npy'),new_class_label)
def main(args):

    num_class = 100
    if args.dataset == 'Food':
        num_class = 101
        alpha = 0.8
    elif args.dataset == 'CUB':
        num_class = 200
        alpha = 0.8
    elif args.dataset == 'CIFAR-10':
        num_class = 10
        alpha = 0.9
    elif args.dataset == 'CIFAR-100':
        num_class = 100
        alpha = 0.8
    elif args.dataset == 'ImageNet':
        num_class = 1000
        alpha = 0.7
    elif args.dataset == 'Flower':
        num_class = 102
        alpha = 0.8
    num_concept = num_class * int(args.time_concept)
    mi = cal_mi(args.dataset, args.shot,args.learn,alpha)
    select_concepts(mi,args.dataset, args.shot, num_class, num_concept,args.learn)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('shot', type=str)
    parser.add_argument('time_concept', type=str)
    parser.add_argument('learn', type=str)
    args = parser.parse_args()
    main(args)
