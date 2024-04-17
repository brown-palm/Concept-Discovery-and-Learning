import time

import requests
import json
from tqdm import tqdm
import argparse
import os
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "your openai api key"
}
def generate_prompt(category_name, concept):
    return f"""Please just answer "yes" or "no". Does the {category_name} usually have the visual attribute "{concept}"?"""

def send_request(category,concept):
    data = {
                 "model": "gpt-3.5-turbo-1106",
                 "messages": [
                     {"role": "user",
                      "content": generate_prompt(category,concept)}
                 ]
             }
    session = requests.Session()

    response = session.post(url, headers=headers, data=json.dumps(data)).json()

    session.close()
    if response['choices'][0]['message']['content'].lower() == 'yes' or response['choices'][0]['message'][
        'content'].lower() == 'yes.':
        return 1
    else:
        return 0
def main(args):
    with open('../concept_discovery/new_concepts.json','r') as fp:
        concepts = json.load(fp)
    with open(os.path.join('..',args.dataset,'category.json'),'r') as fp:
        categories = json.load(fp)
    category2concept = {}
    for category in categories:
        category2concept[category] = []
        for concept in concepts:
            if send_request(category,concept) == 1:
                category2concept[category].append(concept)
    with open(os.path.join('..',args.dataset,'new_concept_f.json'),'w') as fw:
        json.dump(category2concept,fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    main(args)

