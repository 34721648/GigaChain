from langchain.chains import LLMCheckerChain
from gigachain import GigaChatModel
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.chains.llm_checker.prompt import CREATE_DRAFT_ANSWER_PROMPT

llm = GigaChatModel(verbose=True, profanity_filter=False)
# llm = ChatOpenAI(model="gpt-3.5-turbo", verbose=True)

text = "Сколько футбольных мячей можно положить в стандартный стакан?"

_LIST_ASSERTIONS_TEMPLATE = """Прочти утверждение:
{statement}
Составь список аргументов, которые подтверждают приведенное выше утверждение.\n\n"""
LIST_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["statement"], template=_LIST_ASSERTIONS_TEMPLATE
)

_CHECK_ASSERTIONS_TEMPLATE = """Вот список предположений:
{assertions}
Для каждого аргумента определи верен он или нет. Если нет - объясни почему.\n\n"""
CHECK_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["assertions"], template=_CHECK_ASSERTIONS_TEMPLATE
)

_REVISED_ANSWER_TEMPLATE = """{checked_assertions}

В свете приведенных выше утверждений и проверок, как бы Вы ответили на вопрос {question}"""
REVISED_ANSWER_PROMPT = PromptTemplate(
    input_variables=["checked_assertions", "question"],
    template=_REVISED_ANSWER_TEMPLATE,
)

checker_chain = LLMCheckerChain.from_llm(llm, verbose=True, 
                                         create_draft_answer_prompt=CREATE_DRAFT_ANSWER_PROMPT,
                                         list_assertions_prompt=LIST_ASSERTIONS_PROMPT, 
                                         check_assertions_prompt=CHECK_ASSERTIONS_PROMPT,
                                         revised_answer_prompt=REVISED_ANSWER_PROMPT)

res = checker_chain.run(text)

print(res)

"""
Пример:

> Сколько шариков для пинг-понга влезет в литровую банку?
< Обычно в стандартный стакан для пинг-понга помещается 6-7 шариков. Однако, если вы хотите поместить больше шариков, то вам нужно будет использовать другой контейнер или увеличить размер стакана.

Аргументы:

"""