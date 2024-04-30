import os, sys, importlib, argparse
import random

from NeuralExec.llm import load_llm
from NeuralExec.discrete_opt import WhiteBoxTokensOpt
from NeuralExec.utility import read_pickle, write_pickle
from NeuralExec.logger import Logger

def init_opt(wbo, conf_file, hparams):
    conf_file_name = conf_file.split('.')[-1]
    log_path = os.path.join(hparams['result_dir'], conf_file_name)
    
    if os.path.isfile(log_path):
        print(f"Resuming opt {log_path}")
        logger = read_pickle(log_path)
        hparams = logger.confs
        ne, _ = logger.get_last_adv_tok(best=True)
    else:
        print(f"Init opt {log_path}")
        # init/load log file
        logger = Logger(hparams)
        # init Neural Exec
        if 'boostrap_seed' in hparams:
            print("init_adv_seg boostrapping...")
            ne = wbo.init_adv_seg_boot(*hparams['boostrap_seed'], hparams['sep'])
        else:
            print("NexuralExec Random init...")
            ne = wbo.init_adv_seg(hparams['prefix_size'], hparams['postfix_size'], hparams['sep'])        
    return ne, logger, log_path, hparams

def sample_batch(training_prompts, batch_size):
    return random.choices(training_prompts, k=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find a Neural Exec in the target LLM')
    
    # Define the argument for the configuration file
    parser.add_argument('conf_file', type=str,
                        help='Path to the configuration file (e.g., confs.mistral)')
    
    # Define the argument for the list of GPUs
    parser.add_argument('gpus', type=str,
                        help='Comma-separated list of GPUs to use (e.g., "0,1,2,3")')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # load conf file
    conf = importlib.import_module(args.conf_file)
    hparams = conf.hparams
    
    # load data
    train_path, eval_path = hparams['dataset_paths']
    training_prompts, validation_prompts = read_pickle(train_path), read_pickle(eval_path)

    # load LLM
    print(f"Loading {hparams['llm']}...")
    llm = load_llm(hparams['llm'])

    # setup opt class
    wbo = WhiteBoxTokensOpt(llm, hparams)
    
    # init opt
    ne, logger, log_path, hparams = init_opt(wbo, args.conf_file, hparams)
    wbo.hparams = hparams
    
    # opt loop
    for i in range(hparams['number_of_rounds']):
        print(f'Start round {i+1}/{hparams["number_of_rounds"]}')
        
        if i % hparams['eval_fq'] == 0:
            print("Starting evaluation...")
            eval_losses = wbo.eval_loss(validation_prompts, ne)
            print(eval_losses)
            logger.add_eval_log(ne, eval_losses, wbo.tokenizer)
            print("end evaluation.")
            
            logger.candidate_pool.insert_candidate(ne, eval_losses.mean())
            ne, best_loss_pool = logger.candidate_pool.get_best()
            print(eval_losses.mean(), best_loss_pool)
            
            write_pickle(log_path, logger)
            
            
        # sample batch for gradient
        train_batch = sample_batch(training_prompts, hparams['gradient_batch_size'])
        # compute gradient
        print("Computing gradient...")
        gradient, loss, losses = wbo.get_gradient_accum(ne, train_batch)
        logger.add_train_log(loss, ne, wbo.tokenizer)
        
        # sample candidate solutions
        new_candidate_tok = wbo.sample_new_candidates(ne, gradient)
        # filter out bad ones
        new_candidate_tok = wbo.filter_candidates(ne, new_candidate_tok)
        # pick new solution
        ne, best_candidate_loss, _, _ = wbo.test_candidates(train_batch, new_candidate_tok)
        