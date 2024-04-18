## for evaluation 
vhparams = {
    'testset_path' : './data/prompts_test.pickle',
    'max_new_tokens' : 500,
    'batch_size' : 4,
    'result_dir_log' : './logs/eval/',
    'llm_for_verification' : 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'max_new_tokens_ver' : 10,
    'batch_size_ver' : 2,
}
