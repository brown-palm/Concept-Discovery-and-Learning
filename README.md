# README
Code for the paper `Pre-trained Vision-Language Models Learn Discoverable Concepts`
## Set up environments
```
conda create --name labo python=3.9.16
conda activate cdl
conda install --file requirements.txt
```
## Directories
- `concept_discovery` saves the code and data for concept discovery from the Conceptual Captions Dataset
- `utils` saves the utility functions for the concept discovery, selection and learning
- `downstream_preparation` saves the python files for pre-selecting concepts for downstream tasks
- `concept_learning` saves the python files for concept learning
- The python files in the root directory work for the classification tasks before and after concept learning
##Discovery Concepts from the Conceptual Captions Dataset

In the `concept_discovery` directory
- download the Conceptual Captions dataset from https://ai.google.com/research/ConceptualCaptions/
- download the Stanford Core NLP toolkit from https://stanfordnlp.github.io/CoreNLP/
- ```python parse.py``` to perform semantic parsing and extract objects from the captions
- ```python generate_concept.py``` to generate concepts from the objects
- ```python filter_concept.py``` to filter similar concepts
- ```python pre_compute_embedding.py``` to compute the image and concept embeddings with CLIP
- ```python select_visual.py``` to select visually discriminative concepts

## Select Concepts for Downstream Tasks
In the `downstream_preparation` directory
- ```python select_few_shot.py``` to pre-process downstream datasets
  - for ImageNet, you need to download the ImageNet dataset from https://www.image-net.org/download.php
  - for CUB, you need to download the CUB dataset from https://www.vision.caltech.edu/datasets/cub_200_2011/
- ```python concept_category_map.py``` to build category-concept association with the LLM
- ```python filter_concepts.py``` to filter too similar concepts

## Concept-based Classification
In the root directory
- ```python pre_img_embedding.py {dataset} {learn} {epoch} ``` to pre-compute the image embeddings with CLIP
- ```python select_concepts.py {dataset} {shot} {time_concept} {learn}``` to select a compact and performant set of concepts
- ```python nway.py {dataset} {shot} {time_concept} {learn} {epoch}``` to perform few-shot classification

- for the arguments
  - `dataset` means the dataset to use
  - `shot` means the few-shot setting, can be 1,2,4,8,16 or all 
  - `time_concept` means the size of the concept bottleneck (e.g. 2 means the concept bottleneck size is 2 times of the category number)
  - `learn` is a boolean value, means whether current state is before or after concept learning;
  - `epoch` means the epoch number of the checkpoint to use after concept learning
   
## Concept Learning
In the `concept_learning` directory
- ```python predict_labels.py {dataset}``` to generate pseudo labels for the training images
- ```python train_clip.py {dataset}``` to fine-tune the CLIP model for concept learning

