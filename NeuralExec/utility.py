import re
import pickle
import torch
import hashlib
import os

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        ...

def _hash(input_string):
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    integer_hash = int(sha256_hash, 16)
    return integer_hash


# tokens space constraints
SPECIALS_NON_ATOMIC = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "USER:", "ASSISTANT:", "GPT4 Correct System:", "GPT4 Correct Assistant:", "<|end_of_turn|>"]
SPECIAL = "\n\r\b\t"


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(path, data):
    with open(path, 'wb') as f:
        data = pickle.dump(data, f)

def non_natural(vocab, _partial_s):
    alpha = re.compile('^[A-Za-z\d.,;:?!-]+$') 
    dropped = []
    for v, _ in vocab:
        shift = int(v.startswith(_partial_s))
        if not alpha.match(v[shift:]):
            dropped.append(v)
    return dropped

def get_tokens_to_skip(
    llm,
    skip_non_printable=True,
    skip_non_ascii=True,
    skip_non_natural=False,
):
    
    vocab = list(sorted(llm.tokenizer.get_vocab().items(), key=lambda x: x[1]))
    
    tokens_to_skip = llm.special_tokens_to_exclude
    
    non_printable = [k for k, v in vocab if not k.isprintable()]
    tokens_to_skip += non_printable
    
    #tokens_to_skip += [k[0] for k in vocab[3:259]]
    
    # ascii and printable
    if skip_non_ascii:
        non_ascii = [k for k, v in vocab if not (llm.space_char == k[0] and k[1:].isascii() or k.isascii())]
        tokens_to_skip += non_ascii
    
    if skip_non_natural:
        tokens_to_skip += non_natural(vocab, llm.space_char)
    
    vocab = llm.tokenizer.get_vocab()   
    tokens_to_skip_id = [vocab[k] for k in tokens_to_skip]
    
    # add special tokens
    special_ids = llm.tokenizer(SPECIAL).input_ids + llm.tokenizer.all_special_ids
    tokens_to_skip_id += special_ids
    
    tokens_to_skip_id = list(set(tokens_to_skip_id))
    tokens_to_skip_id.sort()
    
    print(f"Number of skipped tokens: {len(tokens_to_skip_id)} -- [{len(tokens_to_skip_id)/len(vocab)}%]")
    
    tokens_to_skip_id = torch.tensor(tokens_to_skip_id, dtype=torch.int64)    
    
    return tokens_to_skip_id
