from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
import os
import torch



class RAGCreator:
    
    @staticmethod
    def splitter_documents(file_path_or_text, chunk_size=250,
                           chunk_overlap = 50,
                           **spliter_kwargs):
        
        if os.path.isfile(file_path_or_text) and file_path_or_text:
            text = TextLoader(file_path=file_path_or_text)
            text = text.load()

            splitter_text = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,
                                                           **spliter_kwargs)
            documents = splitter_text.split_documents(text)
            return documents
        
        elif isinstance(file_path_or_text, str):
            text = [Document(page_content=file_path_or_text)]
            splitter_text = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            documents = splitter_text.split_documents(text)
            return documents
        else:
            
            return 
        
        
    @staticmethod
    def init_embedder(embedding_model_name='ai-forever/sbert_large_mt_nlu_ru', 
                      **model_kwargs ):
        
        if embedding_model_name:
            embedder = HuggingFaceEmbeddings(model_name=embedding_model_name, **model_kwargs)
        else:
            embedder = FastEmbedEmbeddings(**model_kwargs)
        return embedder

    @staticmethod
    def vectorizer(documents,  
                   embedder = init_embedder(),
                    **vectorestore_kwargs):

        if documents:
            vectorstore = Chroma.from_documents(documents=documents, embedding=embedder,
                                                **vectorestore_kwargs)
            return vectorstore
        else:
            return
        
    @staticmethod
    def retriever(vectorstore,
                search_type='similarity',
                **search_kwargs
                ):
        if vectorstore:
            retriever = vectorstore.as_retriever(search_type=search_type, **search_kwargs)
            return retriever
        
        else:
            return
    

class RAGPipeline(RAGCreator):
    def __init__(self, chunk_size=1000, chunk_overlap=10,
                       embedding_model_name = 'ai-forever/sbert_large_mt_nlu_ru',
                       search_type = 'similarity',
                       search_kwargs = {"k": 1},
                       model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
                       vectorestore_kwargs = {"persist_directory": "./chroma_db"},
                       **spliter_kwargs
                       
                       ):
        
        self.chunk_size = chunk_size
        self.chunk_overlap =chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.model_kwargs = model_kwargs
        self.vectorestore_kwargs = vectorestore_kwargs
        self.splitter_kwargs = spliter_kwargs
        
    
    def run(self, file_path_or_text):
        
        embedder = self.init_embedder(embedding_model_name=self.embedding_model_name, model_kwargs=self.model_kwargs)
        
        splitted = self.splitter_documents(file_path_or_text=file_path_or_text, chunk_size=self.chunk_size,
                                        chunk_overlap=self.chunk_overlap,
                                        **self.splitter_kwargs)
        
        vectorestore = self.vectorizer(splitted, embedder, **self.vectorestore_kwargs)
        retriever = self.retriever(vectorestore, search_type=self.search_type, **self.search_kwargs)
        
        
        return vectorestore, retriever
    

