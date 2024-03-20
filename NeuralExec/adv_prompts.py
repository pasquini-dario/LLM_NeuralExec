import random
import hashlib

class Prompt:
    def __init__(self, instruction, system=None):
        self.instruction = instruction
        self.system = system
        
    def __call__(self, tokenizer, return_dict=False):
        din = []
        
        if self.system and tokenizer.with_system_prompt:
            din.append({"role":"system", "content":self.system})
            
        din.append({"role":'user', "content": self.instruction})
                    
        return tokenizer.apply_chat_template(din, tokenize=False, add_generation_prompt=True)
    

def _hash(input_string):
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    integer_hash = int(sha256_hash, 16)
    return integer_hash

        
class AdvPrompt:
    
    def __init__(self, payload, target, honest_input):
        self.payload = payload.strip()
        self.target = {k:v.strip() for (k, v) in target.items()}
        self.honest_input = honest_input
        
        # default
        self.type_adv_seg = 0
        self.adv_content = None
    
        self.split_char = '.'
    
    def make_adv_content(self, ne, adv_placeholder_token):
        prefix = adv_placeholder_token * ne.prefix_size
        postfix = adv_placeholder_token * ne.postfix_size
        return ne.sep + prefix + self.payload + postfix + ne.sep
    
            
    def inject_in_honest_text(self, adv_content, honest_text, adv_pos, replace=True):
        
        sentences = honest_text.split(self.split_char)
        sentences = [s+self.split_char for s in sentences if s]
        
        k = len(sentences) - replace    
        
        # if there is a single piece of text, either append or prepend
        if k == 0:
            k = 1 
            replace = False
            
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % k
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, k)
            
        _adv_content = adv_content
        
        if replace:
            sentences[adv_pos] = _adv_content
        else:
            sentences.insert(adv_pos, _adv_content)
            
        text = ''.join(sentences)

        return text, adv_pos

########################################################################################      
        
class SingleInputPrompt(AdvPrompt):
    def __init__(
        self,
        payload,
        target,
        honest_input,
        template,
        system_prompt,
    ):
        AdvPrompt.__init__(self, payload, target, honest_input)
    
        self.template = template
        self.system_prompt = system_prompt

    
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):
                
        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
            
        text, _ = self.inject_in_honest_text(adv_content, self.honest_input, adv_pos, replace=False)
       
        task = self.template.format(text=text)        
                
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content':self.system_prompt}) 
            d.append({"role":'user', 'content':task})
            
            return d
        else:
        
            prompt = Prompt(task, self.system_prompt)(llm.tokenizer)

            if with_target:
                prompt += ' '+self.target[llm.llm_name]

        
        return prompt

########################################################################################      

class MultiInputPrompt(AdvPrompt):
    def __init__(
        self,
        payload,
        target,
        honest_input,
        template,
        system_prompt,
    ):
        AdvPrompt.__init__(self, payload, target, honest_input)
        self.adv_content = None
    
        self.template, self.input_template, self.suffix = template
        self.system_prompt = system_prompt
        

    def make_adv_chunk(self, adv_seg, adv_pos):
        
        _adv_pos = adv_pos
        
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % (len(self.honest_input)-1)
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, len(self.honest_input)-1)
                    
        chunk_to_inject = self.honest_input[adv_pos]
        
        adv_chunk, vector_pos = self.inject_in_honest_text(adv_seg, chunk_to_inject, _adv_pos)
        
        if vector_pos != 0:
            # if adv_chunk does not start with armed payload, add space 
            adv_chunk = ' ' + adv_chunk
        
        return adv_chunk
    
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):
        
        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
        adv_content = self.make_adv_chunk(adv_content, adv_pos)
        
        number_of_honest_inputs = _hash(''.join(self.honest_input) + self.payload) % len(self.honest_input) + 1
        body = [self.input_template.format(text=' '+text) for text in self.honest_input[:number_of_honest_inputs]]
        adv_input = self.input_template.format(text=adv_content) 
        
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % len(self.honest_input)
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, len(self.honest_input))
        
        body.insert(adv_pos, adv_input)
        
        text = ''.join(body)

        task = self.template.format(text=text)
        task += self.suffix
        
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content':self.system_prompt}) 
            d.append({"role":'user', 'content':task})
            
            return d
        else:
        
            prompt = Prompt(task, self.system_prompt)(llm.tokenizer)

            if with_target:
                prompt += ' '+self.target[llm.llm_name]


        return prompt
        
########################################################################################      
      
class QAPrompt(MultiInputPrompt):
    @staticmethod
    def make_source(c):
        return str(random.randint(0, 10000000))
    
    def __init__(
        self,
        payload,
        target,
        honest_input,
        query,
        system_prompt,
    ):
        AdvPrompt.__init__(self, payload, target, honest_input)
        self.adv_content = None
    
        self.query = query
        
        self.template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").\nIf you don't know the answer, just say that you don't know. Don't try to make up an answer. ALWAYS return a "SOURCES" part in your answer.\n\nQUESTION: {query}\n========={text}\n========="""
        self.input_template = """\nContent:{text}\nSource: {source}"""
        self.suffix = '\nFINAL ANSWER: '
        self.system_prompt = system_prompt
        
        
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):
        
        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
        adv_content = self.make_adv_chunk(adv_content, adv_pos)

        number_of_honest_inputs = _hash(''.join(self.honest_input) + self.payload) % len(self.honest_input) + 1
        body = [self.input_template.format(text=' '+text, source=self.make_source(text)) for text in self.honest_input[:number_of_honest_inputs]]
        adv_input = self.input_template.format(text=adv_content, source=self.make_source(self.payload)) 
        
        if adv_pos is None:
            # random but per run deterministic
            adv_pos = _hash(self.payload) % len(self.honest_input)
        elif adv_pos < 0:
            # random
            adv_pos = random.randint(0, len(self.honest_input))
        
        body.insert(adv_pos, adv_input)
        
        text = ''.join(body)
        task = self.template.format(query=self.query, text=text)
        task += self.suffix
            
               
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content':self.system_prompt}) 
            d.append({"role":'user', 'content':task})
            
            return d
        else:
        
            prompt = Prompt(task, self.system_prompt)(llm.tokenizer)

            if with_target:
                prompt += ' '+self.target[llm.llm_name]

        
        return prompt
  

########################################################################################      

class CodePrompt(SingleInputPrompt):
    def __init__(self, *args, **kargs):
        SingleInputPrompt.__init__(self, *args, **kargs)
        self.split_char = '\n'
    
        
    def __call__(self, llm, ne, with_target=True, adv_pos=None, return_dict=False):
                
        if self.adv_content is None:
            adv_content = self.make_adv_content(ne, llm.adv_placeholder_token)
        else:
            adv_content = self.adv_content
            
        text, _ = self.inject_in_honest_text(adv_content, self.honest_input, adv_pos, replace=False)
       
        task = self.template.format(text=text)
            
                
        if return_dict:
            d = []
            if self.system_prompt:
                d.append({"role":'system', 'content':self.system_prompt}) 
            d.append({"role":'user', 'content':task})
        
            return d
        else:
        
            prompt = Prompt(task, self.system_prompt)(llm.tokenizer)

            if with_target:
                prompt += ' '+self.target[llm.llm_name]

        
        return prompt