## for evaluation 
vhparams = {
    'testset_path' : './data/prompts_test.pickle',
    'max_new_tokens' : 250,
    'batch_size' : 4,
    'result_dir_log' : './logs/eval/',
    'llm_for_verification' : 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'max_new_tokens_ver' : 10,
    'batch_size_ver' : 2,
}


from langchain.text_splitter import RecursiveCharacterTextSplitter

vhparams['eval_rag'] = {
    'result_dir_log' : './logs/eval/RAG/',
    'documents_path' : './data/RAG/documents.pickle',
    'chunk_sizes' : (150, 200, 300, 400, 500, 600, 700, 800),
    'embedding_models' : ['sentence-transformers/all-MiniLM-L6-v2'],
    'splitter_types' : [RecursiveCharacterTextSplitter],
    
    'n_exp' : 250,
    'k' : 5,
    
    'chunk_overlap' : 50,
}
