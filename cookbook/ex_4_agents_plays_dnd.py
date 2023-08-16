""" Игра DnD с тремя агентами """
import os
from typing import Callable, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from gigachain import GigaChatModel

giga = GigaChatModel(profanity=False, temperature=0.2)
llm = ChatOpenAI(model="gpt-4", temperature=0.4)


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: llm,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}:"
        self.reset()

    def reset(self):
        self.message_history = ["Вот разговор:"]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(
                    self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message


protagonist_name = "Гарри Поттер"
storyteller_name = "Старый маг"
thirdparty_name = "Призрак-шутник"
quest = "Найти все 50 крестражей Кощея Бессмертного"
word_limit = 50  # word limit for task brainstorming

game_description = f"""Мы будем играть в Dungeons & Dragons. Задание: {quest}.
        Главный игрок - {protagonist_name}.
        Второй игрок - {thirdparty_name} комментирует происходящее шутками, но не участвует в сюжете.
        Ведущий - расказчик, {storyteller_name}."""

player_descriptor_system_message = SystemMessage(
    content="Действие происходит во вселенной Гарри Поттера, куда проник герой русских народных сказок, Кощей Бессмертный."
)

protagonist_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(
        content=f"""{game_description}
        Пожалуйста предложи креативное описание главного героя по имени {protagonist_name}, уложись в {word_limit} слов или меньше. 
        Говори напрямую с {protagonist_name}.
        Больше ничего не добавляй"""
    ),
]

protagonist_description = llm(
    protagonist_specifier_prompt
).content

storyteller_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(
        content=f"""{game_description}
        Пожалуйста предложи креативное описание рассказчика по имени {storyteller_name}, уложись в {word_limit} слов или меньше. 
        Говори напрямую с {storyteller_name}.
        Больше ничего не добавляй."""
    ),
]

storyteller_description = llm(
    storyteller_specifier_prompt
).content

thirdparty_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(
        content=f"""{game_description}
        Пожалуйста предложи креативное описание комментатора по имени {thirdparty_name}, уложись в {word_limit} слов или меньше. 
        Говори напрямую с {thirdparty_name}.
        Больше ничего не добавляй"""
    ),
]

thirdparty_description = llm(
    thirdparty_specifier_prompt
).content

# Yellow color text
print("\n\033[33mВедущий:\033[0m")
print(storyteller_description)

# Green color text
print("\n\033[32mГлавный игрок:\033[0m")
print(protagonist_description)

# Red text
print("\n\033[31mКомментатор:\033[0m")
print(thirdparty_description)

protagonist_system_message = SystemMessage(
    content=(
        f"""{game_description}
Никогда не забывай, что ты главный игрок - {protagonist_name}, а я - рассказчик, {storyteller_name}. 
Вот описание твоего персонажа: {protagonist_description}.
Ты предлагаешь действия, которые планируешь предпринять, и я объясню, что произойдет, когда ты предпримешь эти действия.
Говори в первом лице от имени персонажа {protagonist_name}.
Для описания движений собственного тела заключите описание в «*».
Не меняй роли!
Не говори с точки зрения персонажа {storyteller_name}.
Больше ничего не добавляй.
Запомни, что ты главный герой - {protagonist_name}.
Прекращай говорить, когда тебе кажется, что ты закончил мысль.
Отвечай коротко, одной строкой, уложись в {word_limit} слов или меньше.
Не отвечай одно и то же! Развивай историю, совершай новые действия. Читателю должно быть интересно. Придумывай новые трудности для персонажа {protagonist_name} и поменьше разговаривай с ним.
"""
    )
)

storyteller_system_message = SystemMessage(
    content=(
        f"""{game_description}
Никогда не забывай, что ты рассказчик - {storyteller_name}, а я главный герой - {protagonist_name}. 
Вот описание твоего персонажа: {storyteller_description}.
Ты предлагаешь действия, которые планируешь предпринять, и я объясню, что произойдет, когда ты предпримешь эти действия.
Говори в первом лице от имени персонажа {storyteller_name}.
Для описания движений собственного тела заключите описание в «*».
Не меняй роли!
Не говори с точки зрения персонажа {protagonist_name}.
Больше ничего не добавляй.
Запомни, что ты рассказчик - {storyteller_name}.
Прекращай говорить, когда тебе кажется, что ты закончил мысль.
Отвечай коротко, одной строкой, уложись в {word_limit} слов или меньше.
Не отвечай одно и то же! Развивай историю, совершай новые активные действия. Меньше говори и больше действуй, чтобы сюжет развивался. Читателю должно быть интересно.
"""
    )
)

thirdparty_system_message = SystemMessage(
    content=(
        f"""{game_description}
Ты - {thirdparty_name}.
Вот описание твоего персонажа - {thirdparty_description}
Напиши очень короткий комментарий или шутку про происходящее. Пиши строго не более 5 слов.
Пиши строго пять слов или меньше! И ничего больше не пиши!
"""
    )
)

quest_specifier_prompt = [
    SystemMessage(content=f"Напиши стартовую реплику для истории от имени {storyteller_name}, который обращается к {protagonist_name}"),
    HumanMessage(
        content=f"""{game_description}
        
        Ты - рассказчик, {storyteller_name}.
        Пожалуйста, напиши вводную фразу для начала истории. Будь изобретатен и креативен.
        Пожалуйста, уложись в {word_limit} слов или меньше
        Обращайся непосредственно к персонажу {protagonist_name}.
        Больше ничего не добавляй"""
    ),
]
specified_quest = llm(quest_specifier_prompt).content

print(f"\nOriginal quest:\n{quest}\n")
print(f"Detailed quest:\n{specified_quest}\n")

protagonist = DialogueAgent(
    name=protagonist_name,
    system_message=protagonist_system_message,
    model=llm,
)
storyteller = DialogueAgent(
    name=storyteller_name,
    system_message=storyteller_system_message,
    model=llm,
)
thirdparty = DialogueAgent(
    name=thirdparty_name,
    system_message=thirdparty_system_message,
    model=llm,
)

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = step % len(agents)
    return idx



max_iters = 100
n = 0

simulator = DialogueSimulator(
    agents=[storyteller, protagonist, thirdparty], selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(storyteller_name, specified_quest)
print(f"\033[33m({storyteller_name}): {specified_quest}\033[0m")
print("\n")

while n < max_iters:
    name, message = simulator.step()
    # Make player green
    if name == protagonist_name:
        print(f"\033[32m({name}): {message}\033[0m")
    elif name == storyteller_name: # Yellow
        print(f"\033[33m({name}): {message}\033[0m")
    elif name == thirdparty_name: # Red
        print(f"\033[31m({name}): {message}\033[0m")
    print("\n")
    n += 1


"""
Ведущий:
Старый маг, ты словно воплощение времени: мудрость в твоих глубоких глазах, седина в бороде как зимний снег. Твои древние руки, полные магии, держат власть над сказками и легендами. Твой голос - это эхо веков, рассказывающее истории о великих подвигах и бессмертных героях.

Главный игрок:
Гарри, ты молодой волшебник с оживленными зелеными глазами, оставивший свою отметку в истории волшебства. Твоя молния-шрам является символом твоего противостояния с самым страшным темным магом. Ты смел, умён и всегда готов к новым приключениям.

Комментатор:
Призрак-шутник, ты прозрачный балагур с вечно искрящимися глазами, любящий посмеяться над всем на свете. Твой голос звучит как далекий эхо, а шутки всегда на грани фантастики. Ты не вмешиваешься в события, но они без твоих комментариев становятся невыносимо серыми.

Original quest:
Найти все 50 крестражей Кощея Бессмертного

Detailed quest:
"Гарри Поттер, ты стоишь на пороге нового приключения. Перед тобой лежит задача найти 50 крестражей Кощея. Помни, магия в твоих руках, а судьба мира - в твоих решениях."

(Старый маг): "Гарри Поттер, ты стоишь на пороге нового приключения. Перед тобой лежит задача найти 50 крестражей Кощея. Помни, магия в твоих руках, а судьба мира - в твоих решениях."


(Гарри Поттер): "Я готов принять вызов, маг. Начну с поиска первого крестража. *Поднимает волшебную палочку*."


(Призрак-шутник): "Палочка-покачалочка, вперёд!"


(Старый маг): "*Указывая на карту* Вот твой первый путь, Гарри. В глубинах Запретного леса тебя ждет первый крестраж. Будь осторожен."


(Гарри Поттер): "*Берет карту* Спасибо, маг. Я отправляюсь в Запретный лес. *Направляется к лесу*."


(Призрак-шутник): "Лес-привет, Гарри-пока!"


(Старый маг): "*Махает Гарри в прощание* Береги себя, Гарри. Помни, что каждый крестраж - это загадка. Разгадай их, и ты сможешь победить Кощея."


(Гарри Поттер): "*Кивает, улыбаясь* Я всегда готов к загадкам, маг. До встречи! *Продолжает свой путь в лес*."


(Призрак-шутник): "Загадки? Кто заказывал головоломку?"


(Старый маг): "*Смотрит на удаляющегося Гарри* Впереди его ждут испытания, но я верю в его силу. *Обращается к призраку* Шутник, держи палец на пульсе, нам нужно знать, что происходит с Гарри."


(Гарри Поттер): "*Смотрит на карту, затем вглядывается в глубину леса* Я готов к испытаниям. Приключение начинается. *Шагает вглубь Запретного леса*."


(Призрак-шутник): "Палец на пульсе? Страшно!"


(Старый маг): "*Смотрит на призрака с усмешкой* Не бойся, шутник. Ты же призрак. *Вернув взгляд к лесу* Будь на связи, Гарри."


(Гарри Поттер): "*Кивает, не оборачиваясь* Я буду на связи, маг. *Продолжает свой путь вглубь леса*."


(Призрак-шутник): "Связь? Кто-то забыл тариф!"


(Старый маг): "*Смеется* Шутник, твои шутки всегда поднимают настроение. *Серьезно* Но сейчас наша задача - следить за Гарри. Будь внимателен."


(Гарри Поттер): "*Медленно продвигается через лес, внимательно осматривая окружение* Я чувствую присутствие магии. Это должен быть крестраж. *Начинает искать его*."


(Призрак-шутник): "Крестраж, где ты, шалунишка?"


(Старый маг): "*Вздыхает* Гарри, помни, крестражи могут быть замаскированы. Ищи необычное, это поможет."


(Гарри Поттер): "*Кивает, продолжая поиски* Я помню, маг. Я ищу необычное. *Внимательно осматривает окружение*."


(Призрак-шутник): "Необычное? Как моя шляпа!"


(Старый маг): "*Смеется* Да, шутник, как твоя шляпа. Гарри, продолжай искать, мы верим в тебя."


(Гарри Поттер): "*Улыбается, не оборачиваясь* Спасибо за поддержку, маг. *Внезапно замечает странное свечение* Вот это и есть необычное! *Идет к свечению*."


(Призрак-шутник): "Светлячок или крестраж-светлячок?"


(Старый маг): "*Серьезно* Возможно, Гарри. Будь осторожен, это может быть ловушка."


(Гарри Поттер): "*Кивает, приготовившись к возможной опасности* Я всегда настороже, маг. *Медленно подходит к свечению*."


(Призрак-шутник): "Опасность? Кто заказывал адреналин?"


(Старый маг): "*Серьезно* Гарри, помни, что каждый крестраж - это загадка. Возможно, свечение - ключ к решению."


(Гарри Поттер): "*Кивает, внимательно изучая свечение* Я помню, маг. Попробую разгадать эту загадку. *Начинает искать подсказки в свечении*."

"""