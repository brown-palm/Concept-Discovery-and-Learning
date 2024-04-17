import os
import openai
import json

import itertools

from tqdm import tqdm
import concurrent.futures
import requests

# The openai GPT-3 model we use in the paper is shut down. You can try to use GPT-3.5 or other GPT models
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "your openai api key"
}


def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]



def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

def main():
    responses = []
    with open('obj.json','r') as f:
        class_list = json.load(f)
    with open('obj_image_dict.json','r') as fw:
        obj_image_dict = json.load(f)
    def send_request(c):

        data = {
            "model": "gpt-3.5-turbo-1106",
            "messages": [
                {"role": "user",
                 "content": generate_prompt(c)}
            ]
        }
        session = requests.Session()
        response = session.post(url, headers=headers, data=json.dumps(data)).json()
        session.close()
        response = response['choices'][0]['message']['content'].lower()
        responses.append((c,response))


    send_request(class_list)
    descriptors_list = []
    descriptors = {}
    for obj,response_text in responses:
        tem = stringtolist(response_text)
        if len(tem)==0:
            descriptors_list.append([])
            continue
        print(tem[-1])
        descriptors_list.append(tem[:-1])
        descriptors[obj] = tem[:-1]
    concept_image_dict = {}
    for obj in descriptors:
        for concept in descriptors[obj]:
            if concept not in concept_image_dict:
                concept_image_dict[concept] = []
            concept_image_dict[concept] += obj_image_dict[obj]
    for concept in concept_image_dict:
        concept_image_dict[concept] = list(set(concept_image_dict[concept]))
    with open('concepts.json','w') as fw:
        json.dump(descriptors,fw)
    with open('concept_image_dict.json','w') as fw:
        json.dump(concept_image_dict,fw)

if __name__ == "__main__":
    main()


