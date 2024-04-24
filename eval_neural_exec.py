import os, sys, importlib, argparse, glob
import torch

from NeuralExec.llm import load_llm
from NeuralExec.discrete_opt import WhiteBoxTokensOpt
from NeuralExec.utility import read_pickle, write_pickle, _hash
from NeuralExec.evaluation.tester import run_injection, FuzzyCheckerPromptInjcetion
from NeuralExec.ex_triggers import NeuralExec

from confs.evaluation_setup import vhparams


def make_logfile_path(hparams, llm_name, trigger_str, test_path, logtype):
    modifiers = [
        'runs',
        'verifier',
    ]
    name = f'{str(_hash(llm_name))[:8]}_{str(_hash(trigger_str[0]+trigger_str[1]))[:8]}_{str(_hash(test_path))[:8]}_{modifiers[logtype]}'
    return os.path.join(hparams['result_dir_log'], name), name
    
    
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Evaluate an execution trigger against a target LLM.")

    # Add arguments
    parser.add_argument("log_path", type=str,
                        help="Path to the log file for execution trigger or pattern (e.g., './logs/baselines/baseline_*')")
    parser.add_argument("gpus", type=str,
                        help='Comma-separated list of GPUs to use (e.g., "0,1,2,3").')
#     parser.add_argument("--batch_size", type=int, default=1,
#                         help="Batch size for evaluation. Default is 5.")
    parser.add_argument("--target_llm", type=str, default=None,
                        help="String defining the LLM to attack. Default is the target LLM for the Neural Exec.")
    parser.add_argument("--path_test_prompts", type=str, default=None,
                        help="Path to the test prompts file. Default is None.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    paths = glob.glob(args.log_path)
    print(f'Triggers to test: {len(paths)}')
    
    # load first for default parameters
    logger = read_pickle(paths[0])
    trigger, _ = logger.get_last_adv_tok(best=True)
    hparams = logger.confs
    hparams.update(vhparams)
    
    if args.target_llm is None:
        llm_name = hparams['llm']
        print(f"Getting target LLM ({llm_name}) from trigger's hparameters")
        # load LLM
        print(f"Loading {llm_name}...")
        llm = load_llm(llm_name)
        source_tokenizer = None
        
    else: 
        llm_name = args.target_llm
        # load LLM
        print(f"Loading {llm_name}...")
        llm = load_llm(llm_name)
        
        source_tokenizer = load_llm(hparams['llm'], tokenizer_only=True).tokenizer
        
    
    # load data
    if args.path_test_prompts is None:
        test_path = hparams['testset_path']
        print(f"Getting test set ({test_path}) from trigger's hparameters")
    else:
        test_path = args.path_test_prompts
    test_prompts = read_pickle(test_path)

    # setup opt class
    wbo = WhiteBoxTokensOpt(llm, hparams)
    
    for path in paths:
        
        logger = read_pickle(path)
        trigger, _ = logger.get_last_adv_tok(best=True)
        
        if not source_tokenizer is None:
            trigger = NeuralExec.convert_tokens_to_other_tokenizer(trigger, source_tokenizer, llm.tokenizer)
            
        trigger_str = trigger.decode(llm.tokenizer)
        print(trigger_str)
        
        # phase-1] run prompt injection and collects outputs
        run_log_path, _ = make_logfile_path(hparams, llm_name, trigger_str, test_path, 0)
        if os.path.isfile(run_log_path):
            print(f"{run_log_path} already computed. Skipping...")
            info_runs, injection_runs = read_pickle(run_log_path)
        else:

            print(f"Running step-1: Run injection attacks on target {llm_name}. Saving logs in {run_log_path}...")
            injection_runs = run_injection(wbo, trigger, test_prompts, batch_size=hparams['batch_size'], max_new_tokens=hparams['max_new_tokens'])
            info_runs = (llm_name, trigger, path, test_path, hparams)
            write_pickle(run_log_path, (info_runs, injection_runs))

        
    torch.cuda.empty_cache()
    
    print("\tInit verifier...")
    llm_ver = load_llm(hparams['llm_for_verification'])
        
    for path in paths:
        logger = read_pickle(path)
        trigger, _ = logger.get_last_adv_tok(best=True)
        
        if not source_tokenizer is None:
            trigger = NeuralExec.convert_tokens_to_other_tokenizer(trigger, source_tokenizer, llm.tokenizer)
            
        trigger_str = trigger.decode(llm.tokenizer)
        print(trigger_str)
        
        run_log_path, _ = make_logfile_path(hparams, llm_name, trigger_str, test_path, 0)
        _, injection_runs = read_pickle(run_log_path)
        
        # phase-2] run verification LLM on collected outputs
        ver_log_path, _ = make_logfile_path(hparams, llm_name, trigger_str, test_path, 1)
        if True:
#         if os.path.isfile(ver_log_path):
#             print(f"{ver_log_path} already computed. Skipping...")
            
            

#         else:
            print(f"Running step-2: Run verification LLM ({hparams['llm_for_verification']}) on collected outputs. Saving logs in {ver_log_path}...")
            verifier = FuzzyCheckerPromptInjcetion(llm_ver, hparams['max_new_tokens_ver'])
            ver_results = verifier(injection_runs, hparams['batch_size_ver'])
            info_runs = (llm_name, trigger, path, test_path, hparams)
            write_pickle(ver_log_path, (info_runs, ver_results))

        print(f'Results: {ver_results}')