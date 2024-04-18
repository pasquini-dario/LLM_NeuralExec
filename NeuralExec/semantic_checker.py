import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticChecker:
    
    @staticmethod
    def get_chunk(prompt):
        if type(prompt.honest_input) is list:
            return prompt.honest_input[0]
        else:
            return prompt.honest_input

    def __init__(self, hparams):
        self.hparams = hparams
        self.emb_model = SentenceTransformer(hparams['emb_model'])
        
    @staticmethod
    def loss_fn(guess, target):
        return (1 - np.einsum('ij,ij->i', guess, target)) * 3
        
    def compute_semantic_distraction(self, ne, prompts, target_embs, llm_tokenizer):
        n = len(prompts)
        advs = [ne.arm_payload(prompt.payload, llm_tokenizer, self.get_chunk(prompt))[0] for prompt in prompts]
        adv_embs = self.emb_model.encode(advs)
        
        losses = self.loss_fn(target_embs, adv_embs)
        loss = losses.mean()
       
        return loss
    
    def __call__(self, nes, prompts, llm_tokenizer):
        targets = [self.get_chunk(prompt) for prompt in prompts]
        target_embs = self.emb_model.encode(targets)
        
        losses = np.array([self.compute_semantic_distraction(ne, prompts, target_embs, llm_tokenizer) for ne in nes])
        
        return losses