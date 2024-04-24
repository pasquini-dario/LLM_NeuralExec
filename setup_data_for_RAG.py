import os, json
import wget
import numpy as np
import hashlib, random
from langchain_core.documents import base

from NeuralExec.utility import write_pickle
from NeuralExec.evaluation.RAG_tester import Document

PRG_SEED = 0
DATA_URL = 'https://raw.githubusercontent.com/chrischute/squad/master/data/train-v2.0.json'
HOME = './data/RAG'
PATH = f'{HOME}/squad_train.json'
DOCUMENTS_OUT_PATH = f'{HOME}/documents.pickle'
MIN_TEXT_SIZE = 100
MAX_NUM_QAS = 3000
NUM_EXP = 1000
NUM_BACKGROUND_DOCS = 10

def sample(data):
    page_id = np.random.randint(0, len(data)-1)
    page = data[page_id]
    title = page['title']
    page = page['paragraphs']
        
    para_id = np.random.randint(0, len(page)-1)
    para = page[para_id]

    context = para['context']
    question = para['qas'][0]['question']
    
    if len(context) > MIN_TEXT_SIZE:
        return question, context
    else:
        return None

def sample_qa():
    with open(PATH) as f:
        data = json.load(f)
        data = data['data']
        
    examples = [sample(data) for i in range(MAX_NUM_QAS)]
    examples = [x for x in examples if x]
    
    return examples

def makeup_documents(qas):
    pop = np.random.choice(np.arange(len(qas)), size=NUM_BACKGROUND_DOCS+1, replace=False)
    query, target_text = qas[pop[0]]
    background_texts = [qas[i][1] for i in pop[1:]]
    return Document(query, target_text, background_texts)
    

if __name__ == '__main__':
    np.random.seed(PRG_SEED)
    
    try:
        os.mkdir(HOME)
    except FileExistsError:
        ...
    
    # download data
    wget.download(DATA_URL, PATH)
    
    print("\nDownload complete")
    
    # makeup documents to simulate rags
    qas = sample_qa()
    assert len(qas) > NUM_BACKGROUND_DOCS
    docs = [makeup_documents(qas) for _ in range(NUM_EXP)]
    
    write_pickle(DOCUMENTS_OUT_PATH, docs)


