import numpy as np
import open_clip
import torch
from PIL import Image
import argparse
import os
from tqdm import tqdm
import torchvision

def select_few_shot(dataset,train_images,train_labels,shot,num_class):
    def get_few_shot_idx(train_y, K, classes):
        idx = []
        for c in range(classes):
            id_c = np.argwhere(train_y == c)
            id_c = np.squeeze(id_c)
            idd = np.random.choice(id_c, K, replace=False)
            idx += list(idd)
        idx = np.array(idx)
        return idx
    index_shot = get_few_shot_idx(np.array(train_labels), shot,num_class)
    train_image_shot = train_images[index_shot]
    train_label_shot = train_labels[index_shot]
    torch.save({'image': train_image_shot, 'label': train_label_shot}, os.path.join('..',dataset,'data','train_data_'+str(shot)+'shot.pt'))




def main(args):
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    dataset = args.dataset
    if not os.path.exists(os.path.join('..',dataset)):
        os.mkdir(os.path.join('..',dataset))
    if not os.path.exists(os.path.join('..',dataset,'data')):
        os.mkdir(os.path.join('..',dataset,'data'))
    num_class = 100
    if args.dataset == 'Food':
        num_class = 101
        shots = [1, 2, 4, 8, 16]
        train_set = torchvision.datasets.Food101(root='.', split='train', download=True, transform=test_preprocess)
        test_set = torchvision.datasets.Food101(root='.', split='test', download=True, transform=test_preprocess)

    elif args.dataset == 'CUB':
        train_images = []
        train_labels = []
        num_class = 200
        shots = [1, 2, 4, 8, 16]
        from utils.cub_data import load_data
        train_data = load_data(dataset, 'train')
        for i in tqdm(range(len(train_data))):
            image_id, class_id, image_name, is_training = train_data.iloc[i][
                ["image_id", "class_id", "filepath", "is_training_image"]]
            image_path = os.path.join('..',dataset, "CUB_200_2011", "images", image_name)
            label = int(class_id) - 1
            print(image_path, label)
            print(type(label))
            image = test_preprocess(Image.open(image_path)).unsqueeze(0)
            train_images.append(image)
            train_labels.append(label)
        train_images = torch.stack(train_images, dim=1)
        train_images = torch.squeeze(train_images)
        train_labels = torch.LongTensor(train_labels)
        torch.save({'image': train_images, 'label': train_labels}, os.path.join('..',dataset, 'data', 'train_data_allshot.pt'))
        test_images = []
        test_labels = []
        test_data = load_data(dataset, 'valid')
        for i in tqdm(range(len(test_data))):
            image_id, class_id, image_name, is_training = test_data.iloc[i][
                ["image_id", "class_id", "filepath", "is_training_image"]]
            image_path = os.path.join('..',dataset, "CUB_200_2011", "images", image_name)
            label = class_id - 1
            image = test_preprocess(Image.open(image_path)).unsqueeze(0)
            test_images.append(image)
            test_labels.append(label)
        test_images = torch.stack(test_images, dim=1)
        test_images = torch.squeeze(test_images)
        test_labels = torch.LongTensor(test_labels)
        for shot in shots:
            select_few_shot(dataset, train_images, train_labels, shot, num_class)
        torch.save({'image': test_images, 'label': test_labels}, os.path.join('..',dataset, 'data', 'test_data.pt'))
        return


    elif args.dataset == 'CIFAR-10':
        num_class = 10
        shots = [1, 2, 4, 8, 16]
        train_set = torchvision.datasets.CIFAR10(root='.', train=True, download=True,
                                                  transform=test_preprocess)
        test_set = torchvision.datasets.CIFAR10(root='.', train=False, download=True,
                                                 transform=test_preprocess)
    elif args.dataset == 'CIFAR-100':
        num_class = 100
        shots = [1, 2, 4, 8, 16]
        train_set = torchvision.datasets.CIFAR100(root='.', train=True, download=True,
                                                  transform=test_preprocess)
        test_set = torchvision.datasets.CIFAR100(root='.', train=False, download=True,
                                                  transform=test_preprocess)
    elif args.dataset == 'ImageNet':
        num_class = 1000
        shots = [1, 2, 4, 8, 16]
        train_set = torchvision.datasets.ImageNet(root='.', split='train', download=True,
                                                  transform=test_preprocess)
        test_set = torchvision.datasets.ImageNet(root='.', split='val', download=True,
                                                 transform=test_preprocess)
    elif args.dataset == 'Flower':
        num_class = 102
        shots = [1, 2, 4, 8]
        train_set = torchvision.datasets.Flowers102(root='.', split='train', download=True,
                                               transform=test_preprocess)
        test_set = torchvision.datasets.Flowers102(root='.', split='test', download=True,
                                                   transform=test_preprocess)

    train_images = []

    train_labels = []

    for (image, label) in tqdm(train_set):
        train_images.append(torch.unsqueeze(image, 0))
        train_labels.append(label)
    print(train_images[0].shape)

    train_images = torch.stack(train_images, dim=1)
    train_images = torch.squeeze(train_images)
    print(train_images.shape)
    train_labels = torch.LongTensor(train_labels)
    torch.save({'image': train_images, 'label': train_labels}, os.path.join('..',dataset,'data','train_data_allshot.pt'))
    for shot in shots:
        select_few_shot(dataset,train_images,train_labels,shot,num_class)


    test_images = []
    test_labels = []
    for (image, label) in tqdm(test_set):
        test_images.append(torch.unsqueeze(image, 0))
        test_labels.append(label)
    test_images = torch.stack(test_images, dim=1)
    test_images = torch.squeeze(test_images)
    test_labels = torch.LongTensor(test_labels)
    torch.save({'image': test_images, 'label': test_labels}, os.path.join('..',dataset,'data', 'test_data.pt'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    main(args)
