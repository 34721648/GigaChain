from langchain.prompts.chat import (ChatPromptTemplate,
                                    PromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""Используй следующие фрагменты контекста, чтобы ответить на вопрос пользователя.
Если ты не знаешь ответа или ответа на вопрос нет в контексте ниже, просто скажи, что не знаешь, не пытайся придумывать ответ.
----------------
{context}"""),
    HumanMessagePromptTemplate.from_template("{question}"),
])

SUMMARY_STUFF_PROMPT = PromptTemplate(
    template="""Напиши краткое (не более 100 слов) содержимое следующего текста: "{text}". Теперь перескажи тот текст очень кратко.""", input_variables=["text"])
