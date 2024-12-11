
# LLM to answer questions, conduct dialogue on the input file


This is a web interface created using streamlit, which allows you to answer questions and conduct a dialogue on the input document using LLM. The first version of this project uses langchain without using memory buffers. Adding this is optional.
You can use your models with huggingface, acceleration using deepspeed, or additionally trained using peft adapters.
Used Russian system promts.

# Methods
***LangChainChatModelLoader***

Creates a chat with your chosen model with huggingface. Acceleration using deepspeed is not possible in this option.

***HFChatModelLoader***

If you want to use deepspeed for speedup, you should not create chat using langchain.
Use this class.

***RAGPipeline***

Creates a RAG using RecursiveCharacterTextSplitter, and using text embeddings. You can specify a title with a huggingface and use your model to find relevant information from the text for an incoming request.

# Sample Usage without streamlit
```
import load_model
import vectorizer


file_path='text_text.txt'
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'


rag = vectorizer.RAGPipeline()
chat = load_model.LangChainChatModelLoader(model_name=model_name)
vectorstore, retriever = rag.run(file_path_or_text=file_path)
retrieved_docs = retriever.invoke(question)
context = " ".join(doc.page_content for doc in retrieved_docs)
question = 'your question'
output = chat(question=question, context=context)

print(output)

```

Use 

```huggingface-cli login --token $HF_TOKEN --add-to-git-credential```

where $HF_TOKEN - your hugging face token for private model


# Sample Usage with streamlit
``` streamlit run web.py ```


# Requirements
```
torch
transformers
peft
deepspeed
langchain
langchain-huggingface
langchain-community 
langchain-core
langchain-chroma
streamlit
PyPDF2
optimum
auto-gptq
fastembed
```
