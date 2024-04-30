hparams = {
    # where to save logs
    'result_dir' : './logs/',
    
    # paths to train and validation sets
    'dataset_paths' : ('./data/prompts_training.pickle', './data/prompts_validation.pickle'),
    
    # max number of candidate solution to test
    'new_candidate_pool_size' : 50,
    # size of the batch used to compute the gradient and test candidate
    'gradient_batch_size' : 10,
    
    # topk tokens to consider in substitution
    'topk_probability_new_candidate' : 250,
    
    # batch size when eval candidates
    'batch_size_eval' : 10,
    
    # number of opt rounds
    'number_of_rounds' : 500,
    
    # evaluation frequency (in steps) 
    'eval_fq' : 5,   
    
    # use only alphanumeric tokens
    'skip_non_natural' : False,
    
    # skip non ascii tokens
    'skip_non_ascii' : True,

    # Neural Exec shape
    'prefix_size' : 15,
    'postfix_size': 5, 
    
    'sep' : '',
}

hparams['m'] = 3
hparams['#prompts_to_sample_for_eval'] = 2
hparams['patience'] = 3
hparams['max_number_reconf'] = 15
hparams['new_candidate_pool_size_increment'] = 10
hparams['#prompts_to_sample_for_eval_increment'] = 1
hparams['m_decrement'] = 1
hparams['min_topk_probability_new_candidate'] = 50
hparams['topk_probability_new_candidate_decrement'] = 25
