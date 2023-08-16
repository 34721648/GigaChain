"""Пример работы с чатом через langchain"""
from langchain.schema import HumanMessage, SystemMessage

from gigachain import GigaChatModel

chat = GigaChatModel()

messages = [
    SystemMessage(
        content="Ты эмпатичный бот-психолог, который помогает пользователю решить его проблемы."
    )
]

while(True):
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    print("Bot: ", res.content)
