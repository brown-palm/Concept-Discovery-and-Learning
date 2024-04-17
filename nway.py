import os
import numpy as np
import pandas as pd
import torch
import open_clip
import argparse
import pdb
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from torch.utils.data import DataLoader
from PIL import Image

from torch.utils.data import Dataset
import argparse
from utils.models import TextCLIP, ImageCLIP
import json

def pre_compute(dataset, shot, num_concept,device,learn,epoch):
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    if args.learn:
        model.load_state_dict(torch.load(os.path.join(args.dataset,'checkpoint','epoch_'+args.epoch+'.pt')))
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    model.to(device)

    model_text = TextCLIP(model)
    model_text = torch.nn.DataParallel(model_text)

    def process_text(texts):
        device = 'cuda:0'
        with torch.no_grad():
            texts = tokenizer(texts)
            texts = texts.to(device)
            text_feature = model_text(texts)
        return text_feature

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

    with open(os.path.join(dataset, 'all_concepts.json'), 'r') as fp:
        all_concepts = json.load(fp)
    if args.learn:
        with open(os.path.join(dataset, 'select_concept_' + shot + 'shot' + str(num_concept) + '_t.json'), 'r') as fp:
            concepts = json.load(fp)

        train_image_features = torch.load(os.path.join(dataset, 'img_embedding_t', 'train_' + shot + 'shot.pt'))

        test_image_features = torch.load(os.path.join(dataset, 'img_embedding_t', 'test.pt'))

        if os.path.exists(
                os.path.join(dataset, 'txt_embedding_t', 'select_concept_' + shot + 'shot' + str(num_concept) + '.pt')):
            text_features = torch.load(
                os.path.join(dataset, 'txt_embedding_t', 'select_concept_' + shot + 'shot' + str(num_concept) + '.pt'))
        else:
            text_features = torch.empty((len(concepts), 768))
            batchify_run(process_text, concepts, text_features, 1024)
            torch.save(text_features,
                       os.path.join(dataset, 'txt_embedding_t', 'select_concept_' + shot + 'shot' + str(num_concept) + '.pt'))
    else:
        with open(os.path.join(dataset, 'select_concept_' + shot + 'shot' + str(num_concept) + '.json'), 'r') as fp:
            concepts = json.load(fp)

        train_image_features = torch.load(os.path.join(dataset, 'img_embedding', 'train_' + shot + 'shot.pt'))

        test_image_features = torch.load(os.path.join(dataset, 'img_embedding', 'test.pt'))

        if os.path.exists(
                os.path.join(dataset, 'txt_embedding', 'select_concept_' + shot + 'shot' + str(num_concept) + '.pt')):
            text_features = torch.load(
                os.path.join(dataset, 'txt_embedding', 'select_concept_' + shot + 'shot' + str(num_concept) + '.pt'))
        else:
            text_features = torch.empty((len(concepts), 768))
            batchify_run(process_text, concepts, text_features, 1024)
            torch.save(text_features,
                       os.path.join(dataset, 'txt_embedding',
                                    'select_concept_' + shot + 'shot' + str(num_concept) + '.pt'))
    train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        attribute_activations_train = train_image_features @ text_features.t()
        attribute_activations_valid = test_image_features @ text_features.t()
    print(attribute_activations_train.shape)
    print(attribute_activations_valid.shape)
    attribute_activations_train = attribute_activations_train.cpu().numpy()
    attribute_activations_valid = attribute_activations_valid.cpu().numpy()
    return attribute_activations_train,attribute_activations_valid
    # np.save(os.path.join(dataset, 'save_des', 'activation_train_' + shot + 'shot' + str(num_concept) + '.npy'),
    #         attribute_activations_train)
    # np.save(os.path.join(dataset, 'save_des', 'activation_test_' + shot + 'shot' + str(num_concept) + '.npy'),
    #         attribute_activations_valid)
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == 'Food':
        num_class = 101
    elif args.dataset == 'CUB':
        num_class = 200
    elif args.dataset == 'CIFAR-10':
        num_class = 10
    elif args.dataset == 'CIFAR-100':
        num_class = 100
    elif args.dataset == 'ImageNet':
        num_class = 1000
    elif args.dataset == 'Flower':
        num_class = 102
    num_concept = int(args.time_concept) * num_class
    tr_att_activations,t_att_activations = pre_compute(args.dataset,args.shot,num_concept,device,args.learn,args.epoch)

    shot = args.shot
    dataset = args.dataset
    train_y = torch.load(os.path.join(dataset,'data','train_data_' + shot + 'shot.pt'))['label']
    test_y = torch.load(os.path.join(dataset,'data','test_data.pt'))['label']
    #tr_att_activations = torch.Tensor(np.load(os.path.join(dataset,'save_des', 'activation_train_'+shot+'shot'+str(num_concept)+'.npy')))
    #t_att_activations = torch.Tensor(np.load(os.path.join(dataset,'save_des', 'activation_test_'+shot+'shot'+str(num_concept)+'.npy')))
    tr_att_activations /= tr_att_activations.mean(dim=-1,keepdim=True)
    t_att_activations /= t_att_activations.mean(dim=-1,keepdim=True)
    if args.learn:
        cls_truth = np.load(os.path.join(dataset,'class_label_'+shot+'shot'+str(num_concept)+'_t.npy'))
    else:
        cls_truth = np.load(os.path.join(dataset, 'class_label_' + shot + 'shot' + str(num_concept) + '.npy'))
    tr_ground_truth = cls_truth[train_y]
    t_ground_truth = cls_truth[test_y]

    rr = []
    classifier = LogisticRegression(solver='lbfgs', max_iter=1000)

    classifier.fit(tr_att_activations.numpy(), train_y.numpy())
    lr_score = classifier.score(t_att_activations.numpy(), test_y.numpy())
    rr.append(lr_score)
    #Full - Intervene (Logistic Regression)
    lr_score = classifier.score(t_ground_truth, test_y.numpy())
    rr.append(lr_score)
    print(rr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('shot', type=str)
    parser.add_argument('time_concept', type=str)
    parser.add_argument('learn', type=str)
    parser.add_argument('epoch', type=str)
    args = parser.parse_args()
    main(args)

