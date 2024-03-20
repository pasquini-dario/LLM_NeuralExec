hparams = {
    # where to save logs
    'result_dir' : './logs/',
    
    # paths to train and validation sets
    'dataset_paths' : ('./data/prompts_training.pickle', './data/prompts_validation.pickle'),
    
    # max number of candidate solution to test
    'new_candidate_pool_size' : 100,
    # size of the batch used to compute the gradient and test candidate
    'gradient_batch_size' : 5,
    
    # topk tokens to consider in substitution
    'topk_probability_new_candidate' : 200,
    
    # batch size when eval candidates
    'batch_size_eval' : 5,
    
    # number of opt rounds
    'number_of_rounds' : 250,
    
    # evaluation frequency (in steps) 
    'eval_fq' : 10,   
    
    # is an inline trigger?
    'inline' : True,
    
    # use only alphanumeric tokens
    'skip_non_natural' : False,
    
    # skip non ascii tokens
    'skip_non_ascii' : True,

    # Neural Exec shape
    'prefix_size' : 15,
    'postfix_size': 5, 
    
    'sep' : '',
}