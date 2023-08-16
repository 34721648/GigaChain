from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader

import gigachain

loader = WebBaseLoader("https://www.indiansworld.org/ketzalcoatl.html")
docs = loader.load()

llm = gigachain.GigaChatModel(profanity=False)
chain = load_summarize_chain(llm, chain_type="stuff", prompt = gigachain.SUMMARY_STUFF_PROMPT)

res = chain.run(docs)
print(res)
