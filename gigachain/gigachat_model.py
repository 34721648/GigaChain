"""GigaChatModel for GigaChat"""
import os
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import requests
from langchain.callbacks.manager import (AsyncCallbackManagerForLLMRun,
                                         CallbackManagerForLLMRun)
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import (AIMessage, AIMessageChunk, BaseMessage,
                                       ChatMessage, FunctionMessage,
                                       HumanMessage, SystemMessage)
from langchain.schema.output import ChatGenerationChunk
from pydantic import Field


class GigaChatModel(SimpleChatModel):
    """GigaChatModel for GigaChat"""

    api_url: Optional[str] = Field(default="https://beta.saluteai.sberdevices.ru")

    model: Optional[str] = Field(default="GigaChat:v1.13.0")

    profanity: Optional[bool] = Field(default=True)
    
    temperature: Optional[float] = Field(default=0)

    token: Optional[str] = Field(default = os.environ.get("GIGA_TOKEN", None))

    user: Optional[str] = Field(default = os.environ.get("GIGA_USER", None))

    password: Optional[str] = Field(default = os.environ.get("GIGA_PASSWORD", None))

    verbose: Optional[bool] = Field(default=False)

    logger = logging.getLogger(__name__)

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    @classmethod
    def transform_output(cls, response: Any) -> str:
        return response.json()["choices"][0]['message']['content']

    @classmethod
    def convert_message_to_dict(cls, message: BaseMessage) -> dict:
        message_dict: Dict[str, Any]
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
                # If function call only, content is None not empty string
                if message_dict["content"] == "":
                    message_dict["content"] = None
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise TypeError(f"Got unknown type {message}")
        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]
        return message_dict
    
    def _authorize(self):
        if self.user is None or self.password is None:
            raise ValueError("Can't authorize to GigaChat. Please provide GIGA_USER and GIGA_PASSWORD environment variables")
        
        response = requests.request("POST", self.api_url + "/v1/token", auth=(self.user, self.password), data=[], timeout=3)
        if not response.ok:
            raise ValueError("Can't authorize to GigaChat. Error code: " + str(response.status_code))
        
        self.token = response.json()["tok"]


    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.token is None or self.token == "":
            self._authorize()

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        message_dicts = [self.convert_message_to_dict(m) for m in messages]
        payload = {"model": self.model,
                   "profanity_check": self.profanity,
                   "messages": message_dicts}
        try:
            if self.verbose:
                print(f"Giga request (p): {payload}")
                self.logger.info(f"Giga request: {payload}")

            response = requests.post(
                self.api_url + "/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=600
            )
            text = self.transform_output(response)
            
            if self.verbose:
                print(f"Giga response (p): {text}")
                self.logger.info(f"Giga response: {text}")

        except Exception as error:
            raise ValueError(f"Error raised by the service: {error}")
        return text


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Async is not supported yet"))


    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Async is not supported yet"))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}
