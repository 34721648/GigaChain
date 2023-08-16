import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import gigachain

loader = TextLoader("cookbook/data/bicameral.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=gigachain.GigaChatModel(profanity=False), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs = {"prompt": gigachain.prompts.QA_PROMPT})

query = "Кто придумал идею Бикамерального сознания?"
print(query)
while True:
    ans = qa.run(query)
    print(ans + "\n\n")
    query = input("Ваш вопрос: ")