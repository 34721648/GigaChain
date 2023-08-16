"""Пример использования внешнего хаба промптов"""
from langchain.prompts import load_prompt
from langchain.schema import HumanMessage
from gigachain import GigaChatModel

chat = GigaChatModel()
prompt = load_prompt("lc://prompts/hello-world/prompt.yaml")

messages = [
    HumanMessage(
        content=prompt.format(name='ГигаЧат'),
    )
]

print(chat(messages).content)