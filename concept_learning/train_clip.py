import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
import torch.nn as nn
import os
import open_clip
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader


import argparse




class TrainDataset(Dataset):
    def __init__(self, images,labels,tokenizer):
        self.images=images
        self.labels=labels
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img, label = self.images[i], int(self.labels[i])
        return img,label,i

from utils.models import TextCLIP,ImageCLIP
def main(args):
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',
                                                                                     cache_dir='../clip')
    for (name, param) in model.named_parameters():
        if name != 'text_projection' and name != 'visual.proj' and name != 'logit_scale':
            param.requires_grad = False
        if name == 'text_projection':
            print(param)
            print(param.shape)
        if name == 'visual.proj':
            print(param)
            print(param.shape)
        if name == 'logit_scale':
            print(param)
    params = filter(lambda p: p.requires_grad, model.parameters())
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text)
    model_image = torch.nn.DataParallel(model_image)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    batch_size = 256
    max_epoch = 100
    lr = 5e-4
    train_labels = np.load(os.path.join('..',args.dataset,'pred_labels_all.npy'))
    train_labels = torch.LongTensor(train_labels)
    train_images = torch.load(os.path.join('..',args.dataset, 'data', 'train_data_allshot.pt'))['image']
    train_set = TrainDataset(train_images, train_labels, tokenizer)
    device = 'cuda:0'

    model = model.to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=32)
    with open(os.path.join('..',args.dataset,'all_concepts.json'),'r') as fp:
        texts = json.load(fp)
    texts = tokenizer(texts)
    texts = texts.to(device)
    cls_truth = np.load(os.path.join('..',args.dataset,'class_label.npy'))
    cls_truth = torch.Tensor(cls_truth).t()
    cls_truth = cls_truth.to(device)
    if not os.path.exists(os.path.join('..',args.dataset,'checkpoint')):
        os.mkdir(os.path.join('..',args.dataset,'checkpoint'))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch)
    for epoch in range(max_epoch):

        model.train()
        total = 0.
        correct_cupl = 0.
        avg_loss = 0.

        for (images, targets, num) in tqdm(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            image_features = model_image(images)
            text_features = model_text(texts)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            pred = torch.matmul(image_features, text_features.t()) * model.logit_scale
            pred = torch.matmul(pred, cls_truth)

            loss = loss_func(pred, targets)
            loss.backward()
            loss_item = loss.detach().cpu().numpy()
            avg_loss += loss_item
            # convert_models_to_fp32(model)
            optimizer.step()
            total += len(images)

        scheduler.step()
        print('epoch:', epoch, 'epoch loss:', np.mean(avg_loss / total * batch_size))
        if epoch % 10 == 9:
            torch.save(model.state_dict(),os.path.join('..',args.dataset,'checkpoint','epoch_'+str(epoch)+'.pt'))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    main(args)




