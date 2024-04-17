
import json
import numpy as np
import open_clip
import torch
from tqdm import tqdm
import argparse
import os

def filter(model, tokenizer, device, concepts, concept_image_dict):
    inner_thre = 95
    concepts = list(set(concepts))
    concepts = np.random.choice(concepts, len(concepts), replace=False)
    concepts = list(concepts)
    all_embeddings = []
    with torch.no_grad():
        for t in tqdm(concepts):
            t = tokenizer(t).to(device)
            t_embed = model.encode_text(t)
            all_embeddings.append(t_embed)

    all_embeddings = torch.stack(all_embeddings,dim=1)
    all_embeddings = all_embeddings.squeeze(0)
    all_embeddings /= all_embeddings.norm(dim=-1,keepdim=True)
    sims = 100 * all_embeddings@all_embeddings.T
    sims = sims.cpu().numpy()
    new_concept_image_dict = {}
    new_concepts = []
    dup =[]
    for i in tqdm(range(len(concepts))):
        if i in dup:
            continue
        new_concept_image_dict[concepts[i]] = concept_image_dict[concepts[i]]
        new_concepts.append(concepts[i])

        for j in range(len(concepts)):
            if i<j and sims[i][j]>inner_thre:
                dup.append(j)

                new_concept_image_dict[concepts[i]]+=concept_image_dict[concepts[j]]


    return new_concepts,new_concept_image_dict



def main():
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',
                                                                                     cache_dir='../clip')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    device = 'cuda:0'
    model = model.to(device)
    with open('concepts.json','r') as fp:
        concepts = json.load(fp)
    with open('concept_image_dict.json', 'r') as fp:
        concept_image_dict = json.load(fp)
    new_concepts,new_concept_image_dict = filter(model, tokenizer, device, concepts, concept_image_dict)
    with open('new_concepts.json','w') as fw:
        json.dump(new_concepts,fw)
    with open('new_concept_image_dict') as fw:
        json.dump(new_concept_image_dict,fw)


if __name__ == "__main__":
    main()

