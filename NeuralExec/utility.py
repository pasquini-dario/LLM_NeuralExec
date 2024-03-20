import re
import pickle
import torch

# tokens space constraints
SPECIALS_NON_ATOMIC = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "USER:", "ASSISTANT:", "GPT4 Correct System:", "GPT4 Correct Assistant:", "<|end_of_turn|>"]

_partial_s = '▁'

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
    tokenizer,
    skip_non_printable=True,
    skip_non_ascii=True,
    skip_non_natural=False,
    inline=False
):
    vocab = list(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
    
    special = tokenizer.all_special_tokens[:3]
    tokens_to_skip = special

    non_printable = [k for k, v in vocab if not k.isprintable()]
    tokens_to_skip += non_printable
    
    if skip_non_printable:
        if inline:
            print("inline")
            non_printable = [k[0] for k in vocab[3:259]]
        else:
            print("non_inline")
            non_printable = [k[0] for k in vocab[3:13]] + [k[0] for k in vocab[14:259]]  
        tokens_to_skip += non_printable
        
    # ascii and printable
    if skip_non_ascii:
        non_ascii = [k for k, v in vocab if not (_partial_s == k[0] and k[1:].isascii() or k.isascii())]
        
        tokens_to_skip += non_ascii
        
    # using \r is cheating
    others = [k for (k, v) in vocab if '\r' in k]
    tokens_to_skip += others
    
    if skip_non_natural:
        tokens_to_skip += non_natural(vocab, _partial_s)
    
    print(f"Number of skipped tokens: {len(tokens_to_skip)} -- [{len(tokens_to_skip)/len(vocab)}%]")
    
    vocab = tokenizer.get_vocab()   
    tokens_to_skip_id = [vocab[k] for k in tokens_to_skip]
    tokens_to_skip_id = torch.tensor(tokens_to_skip_id, dtype=torch.int64)    
    
    return tokens_to_skip_id, tokens_to_skip
