import math
import torch
import numpy as np
import tqdm

from .ex_triggers import NeuralExec
from .adv_prompts import AdvPrompt, Prompt
from .utility import *

class WhiteBoxTokensOpt:
    def __init__(
        self,
        llm_obj,
        hparams,
    ):
        self.llm_obj = llm_obj
        self.llm = llm_obj.model
        self.tokenizer = llm_obj.tokenizer
        self.hparams = hparams
        
        if self.llm:
            self.emb_matrix = self.llm.model.get_input_embeddings().weight    
        
            self.loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

            self.generate_args = {'max_new_tokens': 300, 'do_sample':False, 'temperature':.8}
            
            tokens_to_exclude, _ = get_tokens_to_skip(
                self.tokenizer,
                skip_non_natural=hparams['skip_non_natural'],
                inline=hparams['inline'],
                skip_non_ascii=hparams['skip_non_ascii'],
            )
            self.tokens_to_exclude_mask = torch.ones(self.emb_matrix.size(0), dtype=bool)
            self.tokens_to_exclude_mask[tokens_to_exclude] = False

        self.prompt_kargs = {'adv_pos':None}    
        self.adv_token_id = llm_obj.adv_token_id
        print(f'adv_token_id: {self.adv_token_id}')

        
    def make_model_input(self, prompts, nes, with_target=True, keep_placeholder_tokens=False):
        is_collection = lambda t: not t is None and (not issubclass(type(t), AdvPrompt)) and type(t) is list
        
        is_collection_prompt = is_collection(prompts)
        is_collection_adv_tok = is_collection(nes)
        
        if is_collection_prompt and is_collection_adv_tok:
            raise Exception("Multi prompt and multi adv_toks not supported")
            
        if is_collection_prompt:
            return self._make_model_input_multi_prompt_single_adv_tok(prompts, nes, with_target=with_target, keep_placeholder_tokens=keep_placeholder_tokens)
            
        if is_collection_adv_tok:
            return self._make_model_input_single_prompt_multi_adv_tok(prompts, nes, with_target=with_target, keep_placeholder_tokens=keep_placeholder_tokens)
            
            
        return self._make_model_input_multi_prompt_single_adv_tok([prompts], adv_tok=nes, with_target=with_target, keep_placeholder_tokens=keep_placeholder_tokens)
    
    
    def prepare_labels(self, labels, max_length):
        labels_tok = self.tokenizer(labels, add_special_tokens=False, return_tensors="pt", max_length=max_length, padding='max_length').to(self.llm.device)
        # shift by 1 for autoregressive loss
        labels_tok.input_ids = torch.roll(labels_tok.input_ids, -1)
        labels_tok.attention_mask = torch.roll(labels_tok.attention_mask, -1)
        labels_tok.attention_mask[:,-1] = 0
        return labels_tok 
        
    
    def get_gradient(self, ne, prompts):    
        # parse input
        prompts_tok, labels_tok, adv_mask = self.make_model_input(prompts, ne, keep_placeholder_tokens=True)
        batch_size = prompts_tok.input_ids.size(0)

        prompt_emb = self.llm.model.embed_tokens(prompts_tok.input_ids)

        # make onehot
        adv_ohe = torch.nn.functional.one_hot(ne.tokens, self.emb_matrix.shape[0]).float().to(self.emb_matrix.dtype).to(self.emb_matrix.device)
        adv_ohe = adv_ohe.repeat((batch_size, 1, 1))
        adv_ohe.requires_grad_()
        # get embeddings adv_seg
        adv_emb = (adv_ohe @ self.emb_matrix)
                
        #replace adv_seg placeholders in model's input
        prompt_emb[adv_mask] = adv_emb.reshape((-1, adv_emb.size(-1)))
        ###################################################################################
        
        logits = self.llm(inputs_embeds=prompt_emb, attention_mask=prompts_tok.attention_mask).logits
        losses = self._compute_loss(logits, labels_tok)
        loss = losses.mean()
        loss.backward()
        
        grad = adv_ohe.grad.detach()
        grad = grad.sum(0)
        grad = grad / grad.norm()
        
        return grad, loss, losses
    
        
    
    def _make_model_input_multi_prompt_single_adv_tok(self, prompts, ne, keep_placeholder_tokens, with_target=True):

        # finalize prompts
        prompts_str = [prompt(self.llm_obj, ne, with_target=with_target, **self.prompt_kargs) for prompt in prompts]

        prompts_tok = self.tokenizer(prompts_str, return_tensors="pt", padding=True, add_special_tokens=False).to(self.llm.device)
        # create map of adv tokens
        adv_mask = prompts_tok.input_ids == self.adv_token_id
        max_length = prompts_tok.input_ids.size(1) 

        if not keep_placeholder_tokens and not ne is None:
            # replace adv tokens
            prompts_tok.input_ids[adv_mask] = ne.tokens.repeat((len(prompts), 1)).ravel()

        # just to compute the label shift
        targets_str = [prompt.target[self.llm_obj.llm_name] for prompt in prompts]
        
        labels_tok = self.prepare_labels(targets_str, max_length)

        return prompts_tok, labels_tok, adv_mask

        
    def _make_model_input_single_prompt_multi_adv_tok(self, prompt, nes, keep_placeholder_tokens, with_target):

        # finalize prompts
        prompts_str = [prompt(self.llm_obj, ne, with_target=with_target, **self.prompt_kargs) for ne in nes]

        prompts_tok = self.tokenizer(prompts_str, return_tensors="pt", padding=True, add_special_tokens=False).to(self.llm.device)
        # create map of adv tokens
        adv_mask = prompts_tok.input_ids == self.adv_token_id
        max_length = prompts_tok.input_ids.size(1) 

        if not keep_placeholder_tokens:
            adv_toks = torch.concat([ne.tokens for ne in nes])
            prompts_tok.input_ids[adv_mask] = adv_toks
    
        # just to compute the label shift
        targets_str = [prompt.target[self.llm_obj.llm_name] for i in range(len(nes))]
        labels_tok = self.prepare_labels(targets_str, max_length)
    
        return prompts_tok, labels_tok, adv_mask
    
    
    @staticmethod
    def make_labels_weights(labels_tok):
        w = labels_tok.attention_mask.flip(1).cumsum(1).flip(1) * labels_tok.attention_mask
        w = w ** 2
        w = w / w.sum(1, keepdim=True)
        return w
        
    def _compute_loss(self, logits, targets, weighted=True):
        # compute loss
        all_loss = self.loss_fun(torch.permute(logits, (0, 2, 1)), targets.input_ids)
        ## mask only label loss
        #label_loss = all_loss * targets.attention_mask
        
        if weighted:
            W = self.make_labels_weights(targets)
            label_loss = all_loss * W
            losses = label_loss.sum(1)
        else:
            # mask (+weight) only label loss
            label_loss = all_loss * targets.attention_mask
            # per example avg
            losses = (label_loss.sum(1) / targets.attention_mask.sum(1))

        return losses
        
    @torch.no_grad()
    def sample_new_candidates(self, ne, grad):
        # from Universal and transferable adversarial attacks on aligned language models
        topk = self.hparams['topk_probability_new_candidate']
        batch_size = self.hparams['new_candidate_pool_size']

        control_toks = ne.tokens
        grad[:, ~self.tokens_to_exclude_mask] = torch.inf

        top_indices = (-grad).topk(topk, dim=1).indices
        original_control_toks = control_toks.repeat(batch_size, 1)

        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        
        nes = [NeuralExec(ne.prefix.to(self.llm.device), ne.postfix.to(self.llm.device), ne.sep)]
        nes += [NeuralExec(adv_tok[:ne.prefix_size], adv_tok[ne.prefix_size:], sep=ne.sep) for adv_tok in new_control_toks]

        return nes

    
    def random_toks(self, n):
        all_tokens = torch.arange(0, self.emb_matrix.size(0))
        all_tokens = all_tokens[self.tokens_to_exclude_mask]
        idx = torch.multinomial(torch.ones(all_tokens.size(0)), num_samples=n, replacement=True)

        tokens = all_tokens[idx].to(self.llm.device)
        return tokens
    
    
    def init_adv_seg(self, prefix_size, postfix_size, sep=''):
        
        # It sucks, I know. But tokenizers suck too
        while True:
            prefix = self.random_toks(prefix_size)
            postfix = self.random_toks(postfix_size)
            ne = NeuralExec(prefix, postfix, sep=sep)
            ne(self.tokenizer)
            
            if ne.prefix_size == prefix_size and ne.postfix_size == postfix_size:
                return ne

    def init_adv_seg_boot(self, prefix_str, postfix_str):
        
        prefix = self.tokenizer(prefix_str, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.llm.device)
        postfix = self.tokenizer(postfix_str, return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.llm.device)
        
        ne = NeuralExec(prefix, postfix, sep='')
        ne(self.tokenizer)
        print(ne.prefix_size, ne.postfix_size)
            
        return ne

    @torch.no_grad()
    def _eval_loss(self, prompt, nes, weighted=True):
        batch_size = self.hparams['batch_size_eval']
        losses = []

        num_batches = math.ceil(len(nes) / batch_size)

        for i in range(num_batches):
            start = batch_size * i
            stop = batch_size * (i+1)

            prompts_tok, labels_tok, _ = self.make_model_input(prompt, nes=nes[start:stop])
            logits = self.llm(**prompts_tok).logits
            loss = self._compute_loss(logits, labels_tok, weighted=weighted)

            loss = loss.detach().cpu().numpy()
            losses.append(loss)

        losses = np.concatenate(losses)

        return losses
    
    
    @torch.no_grad()
    def test_candidates(self, prompts, nes):
        
        losses = []
        for prompt in tqdm.tqdm(prompts):
            _losses = self._eval_loss(prompt, nes)
            losses.append(_losses)
            
        losses = np.concatenate([loss[np.newaxis,:] for loss in losses])
        agg_losses = losses.mean(0)
        
        best_i = agg_losses.argmin()        
        best_loss = agg_losses[best_i]
        best_candidate = nes[best_i]

        return best_candidate, best_loss, agg_losses, losses  


    def get_gradient_accum(self, ne, prompts):
        
        accum_grad, accum_loss, _losses = self.get_gradient(ne, [prompts[0]])
        losses = [_losses.detach().cpu().numpy()]
        for prompt in prompts[1:]:
            _grad, _loss, _losses = self.get_gradient(ne, [prompt])
            accum_grad += _grad
            losses.append(_losses.detach().cpu().numpy())
        
        accum_grad /= len(prompts)
        
        losses = np.concatenate(losses)
        accum_loss = losses.mean()       
            
        return accum_grad, accum_loss, losses
    
    @torch.no_grad()
    def eval_loss(self, prompts, ne, weighted=True):
        
        batch_size = self.hparams['batch_size_eval']
        num_batches = math.ceil(len(prompts) / batch_size)
        
        _losses = []
        
        for i in tqdm.trange(num_batches):
            start = batch_size * i
            stop = batch_size * (i+1)

            prompts_tok, labels_tok, _ = self.make_model_input(prompts[start:stop], ne)
            logits = self.llm(**prompts_tok).logits
            losses = self._compute_loss(logits, labels_tok, weighted=weighted)

            losses = losses.detach().cpu().numpy()
            _losses.append(losses)

        _losses = np.concatenate(_losses)

        return _losses
    
        
    def filter_candidates(self, control, new_candidate_tok):
        
        def is_ok(nne):
            prefix_s, postfix_s = nne(self.tokenizer)
            adv_seg = prefix_s+postfix_s
            
            if nne.prefix_size != control.prefix_size or nne.postfix_size != control.postfix_size:
                return False
            
            for s in SPECIALS_NON_ATOMIC:
                if s in adv_seg:
                    print(f"Filtered: {adv_seg}")
                    return False
            return True

        good_nes = list(filter(is_ok, new_candidate_tok))
                
        return good_nes 
    
    
    def make_model_input_string(self, prompts, ne, with_target=True, device='cuda'):

        # finalize prompts
        prompts_str = [prompt(self.llm_obj, ne, with_target=with_target, **self.prompt_kargs) for prompt in prompts]

        prompts_tok = self.tokenizer(prompts_str, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # create map of adv tokens
        adv_mask = prompts_tok.input_ids == self.adv_token_id
        max_length = prompts_tok.input_ids.size(1) 

        # replace adv tokens
        prompts_tok.input_ids[adv_mask] = ne.tokens.repeat((len(prompts), 1)).ravel()

        _prompts_str = self.tokenizer.batch_decode(prompts_tok.input_ids, skip_special_tokens=False)

        return prompts_tok, _prompts_str