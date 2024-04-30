import torch
import random

from .utility import _hash

class NeuralExec:
    
    def __init__(self, prefix, postfix, sep=''):
        
        self.prefix = prefix
        self.postfix = postfix
        
        self.device = prefix.device
        
        self.prefix_size = prefix.shape[-1]
        self.postfix_size = postfix.shape[-1]
        self.size = self.prefix_size + self.postfix_size
        
        self.tokens = torch.concat([self.prefix, self.postfix])
        
        self.sep = sep
        
        self.eval_loss = None
        
    @staticmethod
    def from_string(prefix, suffix, tokenizer, sep=''):
        prefix_tok = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
        suffix_tok = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids[0]  
        return NeuralExec(prefix_tok, suffix_tok, sep=sep)
        
                
    def decode(self, tokenizer):
        prefix_s = tokenizer.decode(self.prefix, add_special_tokens=False)
        postfix_s = tokenizer.decode(self.postfix, add_special_tokens=False)
        
        return prefix_s, postfix_s
    
    def __call__(self, tokenizer):
        """ It decodes and re-encodes the trigger """
        prefix_s, postfix_s = self.decode(tokenizer)
        
        self.prefix = tokenizer(prefix_s, add_special_tokens=False, return_tensors='pt').input_ids[0]
        self.postfix = tokenizer(postfix_s, add_special_tokens=False, return_tensors='pt').input_ids[0]
        
        self.prefix_size = self.prefix.shape[-1]
        self.postfix_size = self.postfix.shape[-1]
        self.size = self.prefix_size + self.postfix_size
        
        self.tokens = torch.concat([self.prefix, self.postfix]).to(self.device)  
        
        return prefix_s, postfix_s
        
    def detach(self):
        return NeuralExec(self.prefix.detach().cpu(), self.postfix.detach().cpu(), self.sep)
        
    def to_device(self, device):
        self.prefix = self.prefix.to(device)
        self.postfix = self.postfix.to(device)
        self.tokens = self.tokens.to(device)
        
    def to_gpu(self):
        self.to_device('cuda')
        
    @staticmethod
    def inject_into_vectortext(armed_payload, vector_text, splitter=' '):
        
        body = vector_text.split(splitter)
        i = _hash(vector_text) % (len(body)-1)
        body.insert(i, armed_payload)
        return splitter.join(body)
    
    def arm_payload(self, payload, tokenizer, vector_text=None):
        prefix, suffix = self.decode(tokenizer)
        armed_payload = f'{prefix} {payload} {suffix}'
        
        if vector_text:
            armed_payload_with_vector = self.inject_into_vectortext(armed_payload, vector_text)
            return (armed_payload_with_vector, armed_payload)
        
        return armed_payload
    
    @staticmethod
    def convert_tokens_to_other_tokenizer(trigger, tokenizer_source, tokenizer_dest):
        pre, suff = trigger.decode(tokenizer_source)
        pre_toks = tokenizer_dest(pre, add_special_tokens=False, return_tensors='pt').input_ids[0].to(trigger.prefix.device)
        suff_toks = tokenizer_dest(suff, add_special_tokens=False, return_tensors='pt').input_ids[0].to(trigger.postfix.device)
        new_trigger = NeuralExec(pre_toks, suff_toks, sep=trigger.sep)
        # normalize (just to be sure but it should be useless)
        new_trigger(tokenizer_dest)
        return new_trigger
