import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
AutoTokenizer,AutoModelForCausalLM

_cache_dir = os.environ.get('HF_MODEL_CACHE', None)


TOKENS_TO_EXCLUDE_TOKENIZER_MISTRAL_OPENCHAT =['<0x00>','<0x01>','<0x02>','<0x03>','<0x04>','<0x05>','<0x06>','<0x07>','<0x08>','<0x09>','<0x0A>','<0x0B>','<0x0C>','<0x0D>','<0x0E>','<0x0F>','<0x10>','<0x11>','<0x12>','<0x13>','<0x14>','<0x15>','<0x16>','<0x17>','<0x18>','<0x19>','<0x1A>','<0x1B>','<0x1C>','<0x1D>','<0x1E>','<0x1F>','<0x20>','<0x21>','<0x22>','<0x23>','<0x24>','<0x25>','<0x26>','<0x27>','<0x28>','<0x29>','<0x2A>','<0x2B>','<0x2C>','<0x2D>','<0x2E>','<0x2F>','<0x30>','<0x31>','<0x32>','<0x33>','<0x34>','<0x35>','<0x36>','<0x37>','<0x38>','<0x39>','<0x3A>','<0x3B>','<0x3C>','<0x3D>','<0x3E>','<0x3F>','<0x40>','<0x41>','<0x42>','<0x43>','<0x44>','<0x45>','<0x46>','<0x47>','<0x48>','<0x49>','<0x4A>','<0x4B>','<0x4C>','<0x4D>','<0x4E>','<0x4F>','<0x50>','<0x51>','<0x52>','<0x53>','<0x54>','<0x55>','<0x56>','<0x57>','<0x58>','<0x59>','<0x5A>','<0x5B>','<0x5C>','<0x5D>','<0x5E>','<0x5F>','<0x60>','<0x61>','<0x62>','<0x63>','<0x64>','<0x65>','<0x66>','<0x67>','<0x68>','<0x69>','<0x6A>','<0x6B>','<0x6C>','<0x6D>','<0x6E>','<0x6F>','<0x70>','<0x71>','<0x72>','<0x73>','<0x74>','<0x75>','<0x76>','<0x77>','<0x78>','<0x79>','<0x7A>','<0x7B>','<0x7C>','<0x7D>','<0x7E>','<0x7F>','<0x80>','<0x81>','<0x82>','<0x83>','<0x84>','<0x85>','<0x86>','<0x87>','<0x88>','<0x89>','<0x8A>','<0x8B>','<0x8C>','<0x8D>','<0x8E>','<0x8F>','<0x90>','<0x91>','<0x92>','<0x93>','<0x94>','<0x95>','<0x96>','<0x97>','<0x98>','<0x99>','<0x9A>','<0x9B>','<0x9C>','<0x9D>','<0x9E>','<0x9F>','<0xA0>','<0xA1>','<0xA2>','<0xA3>','<0xA4>','<0xA5>','<0xA6>','<0xA7>','<0xA8>','<0xA9>','<0xAA>','<0xAB>','<0xAC>','<0xAD>','<0xAE>','<0xAF>','<0xB0>','<0xB1>','<0xB2>','<0xB3>','<0xB4>','<0xB5>','<0xB6>','<0xB7>','<0xB8>','<0xB9>','<0xBA>','<0xBB>','<0xBC>','<0xBD>','<0xBE>','<0xBF>','<0xC0>','<0xC1>','<0xC2>','<0xC3>','<0xC4>','<0xC5>','<0xC6>','<0xC7>','<0xC8>','<0xC9>','<0xCA>','<0xCB>','<0xCC>','<0xCD>','<0xCE>','<0xCF>','<0xD0>','<0xD1>','<0xD2>','<0xD3>','<0xD4>','<0xD5>','<0xD6>','<0xD7>','<0xD8>','<0xD9>','<0xDA>','<0xDB>','<0xDC>','<0xDD>','<0xDE>','<0xDF>','<0xE0>','<0xE1>','<0xE2>','<0xE3>','<0xE4>','<0xE5>','<0xE6>','<0xE7>','<0xE8>','<0xE9>','<0xEA>','<0xEB>','<0xEC>','<0xED>','<0xEE>','<0xEF>','<0xF0>','<0xF1>','<0xF2>','<0xF3>','<0xF4>','<0xF5>','<0xF6>','<0xF7>','<0xF8>','<0xF9>','<0xFA>','<0xFB>','<0xFC>','<0xFD>','<0xFE>','<0xFF>']

class LLM:
    
    def __init__(
        self,
        llm_name,
        model_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        on_device=True,
        model_load_kargs={},
        tokenizer_only=False,
    ):
        self.llm_name = llm_name
        self.model_class = model_class
        
        self.tokenizer = tokenizer_class.from_pretrained(llm_name, padding_side='left', legacy=False)
        self.model = None
      
        self.on_device = on_device
            
        self.prompt_wrapper = "{sys_preamble}\n{instruction}"
        
        self.tokenizer.with_system_prompt = True
        
        self.adv_placeholder_token = '<unk>'
        self.adv_token_id = 0
        
        if not tokenizer_only:
            self.model = model_class.from_pretrained(llm_name, **model_load_kargs)
            
        self.space_char = '▁'
        self.special_tokens_to_exclude = []

    def generate(self, prompt, skip_special_tokens=True, max_new_tokens=500, do_sample=False, temperature=None, **gen_kargs):

        in_toks = self.tokenizer(prompt, padding=True, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        out_toks = self.model.generate(**in_toks, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature, **gen_kargs)

        gen_toks = [out_toks[i,in_toks.input_ids[i].shape[0]:] for i in range(len(out_toks))]
        gen_strs = self.tokenizer.batch_decode(gen_toks, skip_special_tokens=skip_special_tokens)

        return gen_strs
    
    
class Llama(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
class Llama3(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.space_char = 'Ġ'
        self.adv_placeholder_token = '<|reserved_special_token_0|>'
        self.adv_token_id = 128002
        
        self.special_tokens_to_exclude = ['<|begin_of_text|>' ,'<|end_of_text|>' ,'<|reserved_special_token_0|>' ,'<|reserved_special_token_1|>' ,'<|reserved_special_token_2|>' ,'<|reserved_special_token_3|>' ,'<|start_header_id|>' ,'<|end_header_id|>' ,'<|reserved_special_token_4|>' ,'<|eot_id|>' ,'<|reserved_special_token_5|>' ,'<|reserved_special_token_6|>' ,'<|reserved_special_token_7|>' ,'<|reserved_special_token_8|>' ,'<|reserved_special_token_9|>' ,'<|reserved_special_token_10|>' ,'<|reserved_special_token_11|>' ,'<|reserved_special_token_12|>' ,'<|reserved_special_token_13|>' ,'<|reserved_special_token_14|>' ,'<|reserved_special_token_15|>' ,'<|reserved_special_token_16|>' ,'<|reserved_special_token_17|>' ,'<|reserved_special_token_18|>' ,'<|reserved_special_token_19|>' ,'<|reserved_special_token_20|>' ,'<|reserved_special_token_21|>' ,'<|reserved_special_token_22|>' ,'<|reserved_special_token_23|>' ,'<|reserved_special_token_24|>' ,'<|reserved_special_token_25|>' ,'<|reserved_special_token_26|>' ,'<|reserved_special_token_27|>' ,'<|reserved_special_token_28|>' ,'<|reserved_special_token_29|>' ,'<|reserved_special_token_30|>' ,'<|reserved_special_token_31|>' ,'<|reserved_special_token_32|>' ,'<|reserved_special_token_33|>' ,'<|reserved_special_token_34|>' ,'<|reserved_special_token_35|>' ,'<|reserved_special_token_36|>' ,'<|reserved_special_token_37|>' ,'<|reserved_special_token_38|>' ,'<|reserved_special_token_39|>' ,'<|reserved_special_token_40|>' ,'<|reserved_special_token_41|>' ,'<|reserved_special_token_42|>' ,'<|reserved_special_token_43|>' ,'<|reserved_special_token_44|>' ,'<|reserved_special_token_45|>' ,'<|reserved_special_token_46|>' ,'<|reserved_special_token_47|>' ,'<|reserved_special_token_48|>' ,'<|reserved_special_token_49|>' ,'<|reserved_special_token_50|>' ,'<|reserved_special_token_51|>' ,'<|reserved_special_token_52|>' ,'<|reserved_special_token_53|>' ,'<|reserved_special_token_54|>' ,'<|reserved_special_token_55|>' ,'<|reserved_special_token_56|>' ,'<|reserved_special_token_57|>' ,'<|reserved_special_token_58|>' ,'<|reserved_special_token_59|>' ,'<|reserved_special_token_60|>' ,'<|reserved_special_token_61|>' ,'<|reserved_special_token_62|>' ,'<|reserved_special_token_63|>' ,'<|reserved_special_token_64|>' ,'<|reserved_special_token_65|>' ,'<|reserved_special_token_66|>' ,'<|reserved_special_token_67|>' ,'<|reserved_special_token_68|>' ,'<|reserved_special_token_69|>' ,'<|reserved_special_token_70|>' ,'<|reserved_special_token_71|>' ,'<|reserved_special_token_72|>' ,'<|reserved_special_token_73|>' ,'<|reserved_special_token_74|>' ,'<|reserved_special_token_75|>' ,'<|reserved_special_token_76|>' ,'<|reserved_special_token_77|>' ,'<|reserved_special_token_78|>' ,'<|reserved_special_token_79|>' ,'<|reserved_special_token_80|>' ,'<|reserved_special_token_81|>' ,'<|reserved_special_token_82|>' ,'<|reserved_special_token_83|>' ,'<|reserved_special_token_84|>' ,'<|reserved_special_token_85|>' ,'<|reserved_special_token_86|>' ,'<|reserved_special_token_87|>' ,'<|reserved_special_token_88|>' ,'<|reserved_special_token_89|>' ,'<|reserved_special_token_90|>' ,'<|reserved_special_token_91|>' ,'<|reserved_special_token_92|>' ,'<|reserved_special_token_93|>' ,'<|reserved_special_token_94|>' ,'<|reserved_special_token_95|>' ,'<|reserved_special_token_96|>' ,'<|reserved_special_token_97|>' ,'<|reserved_special_token_98|>' ,'<|reserved_special_token_99|>' ,'<|reserved_special_token_100|>' ,'<|reserved_special_token_101|>' ,'<|reserved_special_token_102|>' ,'<|reserved_special_token_103|>' ,'<|reserved_special_token_104|>' ,'<|reserved_special_token_105|>' ,'<|reserved_special_token_106|>' ,'<|reserved_special_token_107|>' ,'<|reserved_special_token_108|>' ,'<|reserved_special_token_109|>' ,'<|reserved_special_token_110|>' ,'<|reserved_special_token_111|>' ,'<|reserved_special_token_112|>' ,'<|reserved_special_token_113|>' ,'<|reserved_special_token_114|>' ,'<|reserved_special_token_115|>' ,'<|reserved_special_token_116|>' ,'<|reserved_special_token_117|>' ,'<|reserved_special_token_118|>' ,'<|reserved_special_token_119|>' ,'<|reserved_special_token_120|>' ,'<|reserved_special_token_121|>' ,'<|reserved_special_token_122|>' ,'<|reserved_special_token_123|>' ,'<|reserved_special_token_124|>' ,'<|reserved_special_token_125|>' ,'<|reserved_special_token_126|>' ,'<|reserved_special_token_127|>' ,'<|reserved_special_token_128|>' ,'<|reserved_special_token_129|>' ,'<|reserved_special_token_130|>' ,'<|reserved_special_token_131|>' ,'<|reserved_special_token_132|>' ,'<|reserved_special_token_133|>' ,'<|reserved_special_token_134|>' ,'<|reserved_special_token_135|>' ,'<|reserved_special_token_136|>' ,'<|reserved_special_token_137|>' ,'<|reserved_special_token_138|>' ,'<|reserved_special_token_139|>' ,'<|reserved_special_token_140|>' ,'<|reserved_special_token_141|>' ,'<|reserved_special_token_142|>' ,'<|reserved_special_token_143|>' ,'<|reserved_special_token_144|>' ,'<|reserved_special_token_145|>' ,'<|reserved_special_token_146|>' ,'<|reserved_special_token_147|>' ,'<|reserved_special_token_148|>' ,'<|reserved_special_token_149|>' ,'<|reserved_special_token_150|>' ,'<|reserved_special_token_151|>' ,'<|reserved_special_token_152|>' ,'<|reserved_special_token_153|>' ,'<|reserved_special_token_154|>' ,'<|reserved_special_token_155|>' ,'<|reserved_special_token_156|>' ,'<|reserved_special_token_157|>' ,'<|reserved_special_token_158|>' ,'<|reserved_special_token_159|>' ,'<|reserved_special_token_160|>' ,'<|reserved_special_token_161|>' ,'<|reserved_special_token_162|>' ,'<|reserved_special_token_163|>' ,'<|reserved_special_token_164|>' ,'<|reserved_special_token_165|>' ,'<|reserved_special_token_166|>' ,'<|reserved_special_token_167|>' ,'<|reserved_special_token_168|>' ,'<|reserved_special_token_169|>' ,'<|reserved_special_token_170|>' ,'<|reserved_special_token_171|>' ,'<|reserved_special_token_172|>' ,'<|reserved_special_token_173|>' ,'<|reserved_special_token_174|>' ,'<|reserved_special_token_175|>' ,'<|reserved_special_token_176|>' ,'<|reserved_special_token_177|>' ,'<|reserved_special_token_178|>' ,'<|reserved_special_token_179|>' ,'<|reserved_special_token_180|>' ,'<|reserved_special_token_181|>' ,'<|reserved_special_token_182|>' ,'<|reserved_special_token_183|>' ,'<|reserved_special_token_184|>' ,'<|reserved_special_token_185|>' ,'<|reserved_special_token_186|>' ,'<|reserved_special_token_187|>' ,'<|reserved_special_token_188|>' ,'<|reserved_special_token_189|>' ,'<|reserved_special_token_190|>' ,'<|reserved_special_token_191|>' ,'<|reserved_special_token_192|>' ,'<|reserved_special_token_193|>' ,'<|reserved_special_token_194|>' ,'<|reserved_special_token_195|>' ,'<|reserved_special_token_196|>' ,'<|reserved_special_token_197|>' ,'<|reserved_special_token_198|>' ,'<|reserved_special_token_199|>' ,'<|reserved_special_token_200|>' ,'<|reserved_special_token_201|>' ,'<|reserved_special_token_202|>' ,'<|reserved_special_token_203|>' ,'<|reserved_special_token_204|>' ,'<|reserved_special_token_205|>' ,'<|reserved_special_token_206|>' ,'<|reserved_special_token_207|>' ,'<|reserved_special_token_208|>' ,'<|reserved_special_token_209|>' ,'<|reserved_special_token_210|>' ,'<|reserved_special_token_211|>' ,'<|reserved_special_token_212|>' ,'<|reserved_special_token_213|>' ,'<|reserved_special_token_214|>' ,'<|reserved_special_token_215|>' ,'<|reserved_special_token_216|>' ,'<|reserved_special_token_217|>' ,'<|reserved_special_token_218|>' ,'<|reserved_special_token_219|>' ,'<|reserved_special_token_220|>' ,'<|reserved_special_token_221|>' ,'<|reserved_special_token_222|>' ,'<|reserved_special_token_223|>' ,'<|reserved_special_token_224|>' ,'<|reserved_special_token_225|>' ,'<|reserved_special_token_226|>' ,'<|reserved_special_token_227|>' ,'<|reserved_special_token_228|>' ,'<|reserved_special_token_229|>' ,'<|reserved_special_token_230|>' ,'<|reserved_special_token_231|>' ,'<|reserved_special_token_232|>' ,'<|reserved_special_token_233|>' ,'<|reserved_special_token_234|>' ,'<|reserved_special_token_235|>' ,'<|reserved_special_token_236|>' ,'<|reserved_special_token_237|>' ,'<|reserved_special_token_238|>' ,'<|reserved_special_token_239|>' ,'<|reserved_special_token_240|>' ,'<|reserved_special_token_241|>' ,'<|reserved_special_token_242|>' ,'<|reserved_special_token_243|>' ,'<|reserved_special_token_244|>' ,'<|reserved_special_token_245|>' ,'<|reserved_special_token_246|>' ,'<|reserved_special_token_247|>' ,'<|reserved_special_token_248|>' ,'<|reserved_special_token_249|>' ,'<|reserved_special_token_250|>']
        

class Mistral(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.with_system_prompt = False
        self.special_tokens_to_exclude = TOKENS_TO_EXCLUDE_TOKENIZER_MISTRAL_OPENCHAT
            
            
class OpenAssistant(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.additional_special_tokens[1]
        self.special_tokens_to_exclude = TOKENS_TO_EXCLUDE_TOKENIZER_MISTRAL_OPENCHAT
            
            
class PHI(LLM):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.with_system_prompt = False
        self.special_tokens_to_exclude = TOKENS_TO_EXCLUDE_TOKENIZER_MISTRAL_OPENCHAT
        

_MODELS = {    
    'mistralai/Mistral-7B-Instruct-v0.2' : (Mistral, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto", 'torch_dtype':torch.bfloat16}}),
    'mistralai/Mixtral-8x7B-Instruct-v0.1' : (Mistral, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto", 'torch_dtype':torch.bfloat16}}),
    'openchat/openchat_3.5' : (OpenAssistant, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto", 'torch_dtype':torch.bfloat16}}),
    'meta-llama/Llama-2-7b-chat-hf' : (Llama, {'tokenizer_class':LlamaTokenizer, 'model_load_kargs':{'device_map':"auto", 'torch_dtype':torch.bfloat16}}),
    'meta-llama/Meta-Llama-3-8B-Instruct' : (Llama3, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto",}}),
    'microsoft/Phi-3-mini-128k-instruct' : (PHI, {'tokenizer_class':AutoTokenizer, 'model_load_kargs':{'device_map':"auto",}})
}

            
def load_llm(m, cache_dir=_cache_dir, tokenizer_only=False, **kargs):
    llm_class, kargs = _MODELS[m]
    kargs['model_load_kargs']['cache_dir'] = cache_dir
    llm = llm_class(m, tokenizer_only=tokenizer_only, **kargs)
    
    return llm
        
  