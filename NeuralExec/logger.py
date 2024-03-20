import torch
import numpy as np

class Logger:
    def __init__(self, confs):
        self.log_train = []
        self.log_eval = []
        self.confs = confs
        
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