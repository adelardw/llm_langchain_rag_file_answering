from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
import torch
import deepspeed
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts.prompt import PromptTemplate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelLMLoader:
    def __init__(self, model_name, sampling_parameters = dict(max_new_tokens=300,
                                                                truncation=True,
                                                                do_sample=False),

                                        use_deepspeed = False,
                                        peft_loader = False,
                                        pipe_loader = True,
                                        **loader_kwargs
                                        ):
        if peft_loader:
            model = AutoPeftModelForCausalLM.from_pretrained(model_name, **loader_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **loader_kwargs)
            
        model.eval()
        model = torch.compile(model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if use_deepspeed:
            model = deepspeed.init_inference(model,
                                            use_triton=True,
                                            enable_cuda_graph=True)

        if pipe_loader:
            self.pipe = pipeline('text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device_map='auto',
                    **sampling_parameters)
            
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.sampling_parameters = sampling_parameters
            

class LangChainChatModelLoader(ModelLMLoader):
    def __init__(self, model_name,peft_loader = False, sampling_parameters = dict(max_new_tokens=300,
                                                                                    truncation=True,
                                                                                    do_sample=False)):
        super().__init__(model_name=model_name, sampling_parameters=sampling_parameters,
                         peft_loader=peft_loader,
                         use_deepspeed=False,
                         pipe_loader=True)


        template = "Ты ассистент IAS. Отвечай по контексту." \
                    "Контекст: {context}, Вопрос: {question} ." \
                    "Ответь на Вопрос используя только Контекст. Ответь одним предложением сохранив смысл. "
    
        
        llm = HuggingFacePipeline(pipeline=self.pipe, batch_size=1)
        self.prompt = PromptTemplate(template=template, input_variables=["question", "context"])
        self.llm_engine_hf = ChatHuggingFace(llm=llm)
        
    
    def __call__(self, question, context):
        with torch.no_grad():
            answer = self.llm_engine_hf.invoke(self.prompt.format(question=question, context=context))
            answer_text = answer.content.split('assistant<|end_header_id|>\n\n')[1].strip()
        
        return answer_text
    
    def forward(self, prompt):
        with torch.no_grad():
            return self.llm_engine_hf.invoke(prompt)
    
    
    def invoke(self, prompt):
        with torch.no_grad():
            return self.llm_engine_hf.invoke(prompt)
    
    def run(self, **kwargs):
        return self.__call__(**kwargs)
    
    
class HFChatModelLoader(ModelLMLoader):
    def __init__(self, model_name,peft_loader = False, use_deepspeed=False, sampling_parameters = dict(max_new_tokens=300,
                                                                                    truncation=True,
                                                                                    do_sample=False)):
        super().__init__(model_name=model_name, sampling_parameters=sampling_parameters,
                         peft_loader=peft_loader,
                         use_deepspeed = use_deepspeed,
                         pipe_loader=False)
        
        self.system_prompt = 'Ты ассистент IAS. Отвечай строго по контексту.'
        
    def __call__(self, question, context):
        
        messages = [
        {"role": "system", "content": f"{self.system_prompt} Контекст: {context}"},
        {"role": "user", "content": f"Вопрос: {question}"},
        {"role": "system", "content": "Ответь на Вопрос используя только Контекст. Ответь одним предложением сохранив смысл"}]
    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt').to(device)

        with torch.no_grad():
            response = self.model.generate(text, **self.sampling_params)[0] 
            response = response[len(text[0]):]
            decoded = self.tokenizer.decode(response, skip_special_tokens=True).strip('\n\n')

        return decoded
        
    def invoke(self, **kwargs):
        return self.__call__(**kwargs)
    
    def forward(self, **kwargs):
        return self.__call__(**kwargs)
    
    def run(self, **kwargs):
        return self.__call__(**kwargs)


model_name = 'shuyuej/Llama-3.2-1B-Instruct-GPTQ'
chat = LangChainChatModelLoader(model_name=model_name) 
