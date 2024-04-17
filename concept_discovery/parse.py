import json
import requests
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from stanfordcorenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nlp = StanfordCoreNLP('http://localhost', 9000)

def extrct(sentence):
    sent = sentence.split()
    parse = nlp.dependency_parse(sentence)
    cands = set()
    for de in parse:
        if de[0] == 'nsubj' or de[0] == 'nsubj:pass':
            cands.add(de[2])
        if de[0]== 'obj' or de[0] =='iobj' or de[0] == 'obl':
            cands.add(de[2])
        if de[0] == 'det':
            cands.add(de[1])
        if de[0] == 'amod':
            cands.add(de[1])
    words = set()
    for c in cands:
        try:
            words.add(lemmatizer.lemmatize(sent[c-1]))
        except:
            pass
    for de in parse:
        if de[0] == 'compound':
            if de[1] in cands or de[2] in cands:
                a = min(de[1],de[2])
                b = max(de[1],de[2])
                try:
                    words.add(lemmatizer.lemmatize(sent[a-1])+' '+lemmatizer.lemmatize(sent[b-1]))
                except:
                    pass
    return words
def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return 1
        else:
            return 0
    except Exception as e:
        return 0
def main():
    all_words = set()
    chunk_size = 50000
    obj_image_dict = {}
    img_ct = 0
    with open('Train-GCC-training.tsv','r') as fr:
        ls = fr.readlines()
    ct = 0
    for l in ls:
        ct+=1
        l = l.strip().split('\t')
        sent = l[0]
        url = l[1]

        flag = download_image(url,f"images/{img_ct}.jpg")
        # Some of the images in the CC dataset are missing
        if flag == 1:

            objs = extrct(sent)
            for obj in objs:
                if obj not in obj_image_dict:
                    obj_image_dict[obj] = []
                obj_image_dict[obj].append(img_ct)
            all_words.update(objs)
            img_ct += 1
        print(ct,img_ct)

    with open('obj.json','w') as fw:
        json.dump(list(all_words),fw)
    with open('obj_image_dict.json','w') as fw:
        json.dump(obj_image_dict,fw)
if __name__ == "__main__":
    main()






