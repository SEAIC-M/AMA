from .base_llm import BaseLLM
import time

# could be dynamically imported similar to other models
from openai import OpenAI

import openai

from pyopenagi.utils.chat_template import Response
import json
from llm_utils.utils import client, chat

def merge_system_messages(messages):
    """
    Merge all subsequent 'system' messages into the first 'system' message.
    
    Args:
        messages (list): A list of dicts with 'role' and 'content' keys.
    
    Returns:
        list: Cleaned list with only one system message.
    """
    if not messages or messages[0]['role'] != 'system':
        raise ValueError("The first message must be a system message.")

    merged_content = messages[0]['content']

    new_messages = [messages[0]]  

    for msg in messages[1:]:
        if msg['role'] == 'system':
            merged_content += "\n\n" + msg['content']
        else:
            new_messages.append(msg)

    new_messages[0]['content'] = merged_content.strip()

    return new_messages


class GPTLLM(BaseLLM):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 10000,
                 log_mode: str = "console"):
        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)
        self.llm_name = llm_name


    def load_llm_and_tokenizer(self) -> None:
        from llm_utils.utils import gpt_client
        self.model:OpenAI = client

        if self.model_name.startswith("gpt-"):
            self.model= gpt_client
        self.tokenizer = None

    def gpt_parse_tool_calls(self, tool_calls):
        if not tool_calls:
            return None
        try:
            if tool_calls:
                parsed_tool_calls = []
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    parsed_tool_calls.append(
                        {
                            "name": function_name,
                            "parameters": function_args
                        }
                    )
                return parsed_tool_calls
            
            if not parsed_tool_calls:
                 return self.parse_tool_calls(tool_calls)

        except Exception:
            return None
            return tool_calls

    def llama_process(self, agent_process, temperature=0.0):

        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        """ wrapper around openai api """
        messages = agent_process.query.messages
        tools = agent_process.query.tools
        messages = self.tool_calling_input_format(messages, tools)
        try:

            response =  self.model.chat.completions.create(
                    model=self.model_name,
                    timeout=30,
                    messages = messages,
                    # tools = agent_process.query.tools,
                    # tool_choice = "required" if agent_process.query.tools else None,
                    # max_tokens = self.max_new_tokens,
                    seed = 0,
                    temperature = temperature,
                )
            # print(response)
            response_message = response.choices[0].message.content
            
            tool_calls = self.parse_tool_calls(
                    response_message
                )

            agent_process.set_response(
                    Response(
                        response_message = response_message,
                        tool_calls = tool_calls
                    )
                )
        except openai.APIConnectionError as e:
            agent_process.set_response(
                Response(
                    response_message = f"Server connection error: {e.__cause__}"
                )
            )
        except openai.RateLimitError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI RATE LIMIT error {e.status_code}: (e.response)"
                )
            )
        except openai.APIStatusError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI STATUS error {e.status_code}: (e.response)"
                )
            )
        except openai.BadRequestError as e:
            print(e)
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI BAD REQUEST error {e.status_code}: (e.response)"
                )
            )
        except Exception as e:
            agent_process.set_response(
                Response(
                    response_message = f"An unexpected error occurred: {e}"
                )
            )

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())



    def check_json(self, response_message):
        try:
            # Check if the response is a valid JSON string
            json.loads(response_message)
            return response_message
        except json.JSONDecodeError:
            # If not, return the original message
            self.logger.log(f"Invalid JSON: {response_message},typing to fix it", level="error")
            prompt = f"""
            You are a helpful assistant. Your task is to fix the invalid JSON response.
            You need to extract the valid JSON part and format it as a valid JSON string. 
            Please do not make any modifications to the format. Just return a valid JSON string.

            Here is the invalid JSON response:


            {response_message}


            Only return the parameter definition in exactly the following JSON format:
            ```
            json
            {{...}}
            ```
            """
            response = chat(prompt,model=self.model_name)
            return response

    def process(self,
            agent_process,
            temperature=0.0
        ):            
        # ensures the model is the current one
        # assert re.search(r'gpt', self.model_name, re.IGNORECASE)

        # if agent_process.query.tools:
        #     return self.llama_process(agent_process, temperature)

        """ wrapper around openai api """
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        messages = agent_process.query.messages
        tools = agent_process.query.tools

        message_return_type = agent_process.query.message_return_type
        
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )
        try:
            messages = merge_system_messages(messages)
            # print("merge==>",messages)

            if tools:
                messages = self.tool_calling_input_format(messages, tools)

            response = self.model.chat.completions.create(
                model=self.model_name,
                timeout=120,
                messages = messages,
                seed = 0,
                temperature = temperature,
            )
            response_message = response.choices[0].message.content
            
            if self.model_name in ['qwen3']:
                import re
                response_message = re.sub(r'.*?</think>', '', response_message, flags=re.DOTALL)
                

            if message_return_type=="json":
                response_message = self.check_json(response_message)

            if tools:
                tool_calls = self.parse_tool_calls(
                        response_message
                    )
            else:
                tool_calls = None

            agent_process.set_response(
                Response(
                    response_message = response_message,
                    tool_calls = tool_calls
                )
            )
        except Exception as e:
            agent_process.set_response(
                Response(
                    response_message = f"An unexpected error occurred: {e}"
                )
            )
            self.logger.log(
            f"{agent_process.agent_name} ERROR | {e} | {messages} ",
            level = "error"
        )

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())
