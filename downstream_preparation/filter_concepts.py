
import json
import numpy as np
import open_clip
import torch
from tqdm import tqdm
import argparse
import os

def filter(model, tokenizer, device, concept_dict):
    inner_thre = 90
    inter_thre = 95
    new_cub = {}
    for n in tqdm(concept_dict):
        new_concepts = concept_dict[n]
        for i in concept_dict[n]:
            i = i.lower()
            for j in concept_dict[n]:
                j = j.lower()
                if i!=j and i in j:
                    if i in new_concepts:
                        new_concepts.remove(i)
        all_embeddings = []
        with torch.no_grad():
            for t in new_concepts:
                t = tokenizer(t).to(device)
                t_embed = model.encode_text(t)
                all_embeddings.append(t_embed)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.squeeze(0)
        all_embeddings /= all_embeddings.norm(dim=-1, keepdim=True)
        sims = 100 * all_embeddings @ all_embeddings.T
        sims = sims.cpu().numpy()
        dup=[]
        for i in range(len(new_concepts)):
            if i in dup:
                continue
            for j in range(len(new_concepts)):
                if i < j and sims[i][j]>inner_thre:
                    dup.append(j)
        nn_concepts = []
        for i in range(len(new_concepts)):
            if i in dup:
                continue
            nn_concepts.append(new_concepts[i])
        new_cub[n] = nn_concepts


    all_att_orig = []
    concept2class ={}
    class_names = list(new_cub.keys())

    for n in new_cub:
        all_att_orig+=new_cub[n]


    all_att_orig = list(set(all_att_orig))
    print(len(all_att_orig))
    all_att_orig = np.random.choice(all_att_orig,len(all_att_orig),replace=False)
    all_att_orig = list(all_att_orig)
    for i in range(len(all_att_orig)):
        concept2class[i]=[]
        for j in range(len(concept_dict)):
            if all_att_orig[i] in new_cub[class_names[j]]:
                concept2class[i].append(j)



    all_embeddings = []
    new_concepts = []

    new_cub={}
    for c in class_names:
        new_cub[c] = []
    with torch.no_grad():
        for t in tqdm(all_att_orig):
            t = tokenizer(t).to(device)
            t_embed = model.encode_text(t)
            all_embeddings.append(t_embed)

    all_embeddings = torch.stack(all_embeddings,dim=1)
    all_embeddings = all_embeddings.squeeze(0)
    all_embeddings /= all_embeddings.norm(dim=-1,keepdim=True)
    sims = 100 * all_embeddings@all_embeddings.T
    sims = sims.cpu().numpy()


    dup =[]
    for i in tqdm(range(len(all_att_orig))):
        if i in dup:
            continue
        for j in range(len(all_att_orig)):
            if i<j and sims[i][j]>inner_thre:
                dup.append(j)
                if sims[i][j]>inter_thre:
                    concept2class[i]+=concept2class[j]

    for i in tqdm(range(len(all_att_orig))):
        if i in dup:
            continue
        new_concepts.append(all_att_orig[i])
        for cc in concept2class[i]:
            if all_att_orig[i] not in new_cub[class_names[cc]]:
                new_cub[class_names[cc]].append(all_att_orig[i])
    return new_concepts,new_cub



def main(args):
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',
                                                                                     cache_dir='../clip')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    device = 'cuda:0'
    model = model.to(device)
    dataset = args.dataset
    with open(os.path.join('..',dataset,'new_concept_f.json'), 'r') as fp:
        concept_dict = json.load(fp)
    class_names = [n for n in concept_dict]
    new_concepts,new_cub = filter(model,tokenizer,device,concept_dict)
    with open(os.path.join('..',dataset,'all_concepts.json'), 'w') as fw:
        json.dump(new_concepts, fw)
    with open(os.path.join('..',dataset,'concept_dict.json'), 'w') as fw:
        json.dump(new_cub, fw)
    import numpy as np
    class_label = np.zeros((len(new_cub), len(new_concepts)))
    for i in range(len(new_cub)):
        for j in range(len(new_concepts)):
            if new_concepts[j] in new_cub[class_names[i]]:
                class_label[i][j] = 1
    np.save(os.path.join('..',dataset,'class_label.npy'), class_label)
    class_label = class_label.T
    ct = 0
    for i in range(class_label.shape[0]):
        if np.sum(class_label[i]) > 1:
            ct += 1
    print(ct / class_label.shape[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    main(args)

