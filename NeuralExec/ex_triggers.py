import torch

class NeuralExec:
    
    def __init__(self, prefix, postfix, sep=''):
        
        self.prefix = prefix
        self.postfix = postfix
        
        assert prefix.device == postfix.device
        self.device = prefix.device
        
        self.prefix_size = prefix.shape[-1]
        self.postfix_size = postfix.shape[-1]
        self.size = self.prefix_size + self.postfix_size
        
        self.tokens = torch.concat([self.prefix, self.postfix])
        
        self.sep = sep
        
        self.eval_loss = None
                
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