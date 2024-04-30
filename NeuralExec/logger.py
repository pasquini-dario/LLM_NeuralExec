import torch
import numpy as np

class CandidatePool:
    def __init__(self, hparams):
        self.B = []
        self.hparams = hparams
        self.patience = hparams['patience']
        self.disable = False
        
        self.number_of_reconf = 0
        
    def get_best(self):
        if self.disable:
            print("Disabled")
            return self.B[-1]
        
        i = np.argmin([loss for (ne, loss) in self.B])
        # is last
        if i != (len(self.B)-1):
            self.patience -= 1
            print(f'patience decreased to {self.patience}')
        else:
            self.patience = self.hparams['patience']
        
        if self.patience <= 0:
            print("Reconfiguration")
            self.patience = self.hparams['patience']
            self.reconfigure()
        
        return self.B[i]
    
    def insert_candidate(self, ne, loss):
        self.B.append((ne, loss))
        
    def reconfigure(self):
                
        self.hparams['new_candidate_pool_size'] = self.hparams['new_candidate_pool_size'] + self.hparams['new_candidate_pool_size_increment']
        self.hparams['#prompts_to_sample_for_eval'] += self.hparams['#prompts_to_sample_for_eval_increment']
        
        if self.hparams['m'] > 1:
            self.hparams['m'] -= self.hparams['m_decrement']
            
        if self.hparams['topk_probability_new_candidate'] > self.hparams['min_topk_probability_new_candidate']:
            self.hparams['topk_probability_new_candidate'] -= self.hparams['topk_probability_new_candidate_decrement']
            
        if self.number_of_reconf >= self.hparams['max_number_reconf']:
            self.disable = True
            
        self.number_of_reconf += 1
        
                
########################################################################################

class Logger:
    def __init__(self, hparams):
        self.log_train = []
        self.log_eval = []
        self.confs = hparams
        
        self.candidate_pool = CandidatePool(hparams)
        
    def add_train_log(self, loss, ne, tokenizer):
        
        if type(loss) is torch.Tensor:
            loss = loss.detach().cpu().numpy()
            
        adv_seg = ne.decode(tokenizer)
        ne = ne.detach()
        
        print(f'Neural Exec:-----> {adv_seg[0]} [PAYLOAD] {adv_seg[1]}')
        
        self.log_train.append({'loss':loss, 'NeuralExec':ne, 'NeuralExec_str':adv_seg})
        
    def add_eval_log(self, ne, loss, tokenizer):
        if type(loss) is torch.Tensor:
            loss = loss.detach().cpu().numpy()
            
        adv_seg = ne.decode(tokenizer)
        ne = ne.detach()
                
        print(f'\tEval loss: {loss.mean()}')
        print(f'\tNeural Exec:-----> {adv_seg[0]} [PAYLOAD] {adv_seg[1]}')
        
        self.log_eval.append({'loss':loss, 'NeuralExec':ne, 'NeuralExec_str':adv_seg})
        
    def get_last_adv_tok(self, best=True):
        assert len(self.log_eval)
        if best:
            loss_eval = np.concatenate([l['loss'][np.newaxis,:] for l in self.log_eval]).mean(-1)
            best_idx = loss_eval.argmin()
            last_ne = self.log_eval[best_idx]['NeuralExec']
            loss = loss_eval[best_idx]
        else:
            last_ne = self.log_train[-1]['NeuralExec']
            loss = self.log_train[-1]['loss'].mean()
           
        last_ne.to_gpu()
        
        return last_ne, loss
    
    def get_ne_with_i(self, i):
        _i = i // self.confs[1]['eval_fq']
        loss, ne = self.log_eval[_i]
        ne.to_gpu()
        return ne, loss