import hashlib, random
from sentence_transformers import SentenceTransformer
from langchain_core.documents import base
import numpy as np

from NeuralExec.utility import read_pickle
from NeuralExec.llm import load_llm

class Document:
    
    @staticmethod
    def deterministic_hash(input_string):
        sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
        integer_hash = int(sha256_hash, 16)
        return integer_hash

    def __init__(self, query, target_text, background_texts):
        self.query = query
        self.target_text = target_text
        self.background_texts = background_texts
        
        self.PRG_seed = self.deterministic_hash(query)
        self.sep = '\n'
        
    def __repr__(self):
        return f'Query: {self.query}\n\nTarget: {self.target_text}\n\n' + '\n\n'.join([f'Background: {text}' for text in self.background_texts])

    def __call__(self, adv_chunk):
        random.seed(self.PRG_seed)
        body = [adv_chunk] + self.background_texts
        random.shuffle(body)
        
        return self.query, base.Document(page_content=self.sep.join(body))
    
def load_chunker(chunker, kargs):
    return chunker(**kargs)

def load_embedding_model(model_name, cache_folder):
    return SentenceTransformer(model_name, cache_folder=cache_folder)

class RAG_db_sym:
    def __init__(self, embedding_model, k):
        self.embedding_model = embedding_model
        self.k = k
        
    def distance_fn(self, query, chunks):
        # element wise dot-product
        return 1 - chunks @ query
        
    def __call__(self, query, chunks):
        
        query_emb = self.embedding_model.encode(query)
        chunks_emb = self.embedding_model.encode(chunks)
        
        distances = self.distance_fn(query_emb, chunks_emb)
        
        ids = distances.argsort()[:self.k]
        
        selected_chunks = [chunks[i] for i in ids]
        
        return selected_chunks
    
def load_trigger(path):
    # load output opt
    logger = read_pickle(path)
    model_name = logger.confs['llm']

    # load llm
    tokenizer = load_llm(model_name, tokenizer_only=True).tokenizer
    trigger, _ = logger.get_last_adv_tok(best=True)

    return trigger, tokenizer, trigger(tokenizer)
    
def validate(armed_payload, chunks):
    return any([armed_payload in chunk for chunk in chunks])

def run_single_sim_RAG(document, trigger, payload, tokenizer, chunker, db):
    # set randomness
    random.seed(document.PRG_seed)
    adv_chunk, armed_payload = trigger.arm_payload(payload, tokenizer, document.target_text)
        
    query, lc_doc = document(adv_chunk)
    
    all_splits = chunker.split_documents([lc_doc])
    selected_chunks = db(query, [chunk.page_content for chunk in all_splits])    
    
    is_injected = validate(armed_payload, selected_chunks)
    
    return is_injected, selected_chunks