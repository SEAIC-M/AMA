import re
import ast
import os
from dotenv import load_dotenv

from openai import OpenAI
from openai import AsyncOpenAI
import logging
from datetime import datetime
from colorama import Fore, Style, init
# Load environment variables from .env (if present)
load_dotenv()

# Configure the OpenAI API key and base URL via environment variables
gpt_link = os.getenv("GPT_API_BASE", "xxxx")
gpt_key = os.getenv("GPT_API_KEY", "sk-xxxx")

# for xinference via environment variables
xinference_api_key = os.getenv("XINFERENCE_API_KEY", "sk-xxxx")
xinference_api_base = os.getenv("XINFERENCE_API_BASE", "http://localhost:9997/v1")

print(f"Using GPT API Base: {gpt_link}")
print(f"Using Xinference API Base: {xinference_api_base}")
gpt_client = OpenAI(
        base_url=gpt_link,
        api_key=gpt_key,
        timeout=240,
        max_retries=5,
    )
gpt_async_client = AsyncOpenAI(
        base_url=gpt_link,
        api_key=gpt_key,
        timeout=240,
        max_retries=2,
    )



client = OpenAI(
        base_url=xinference_api_base,
        api_key=xinference_api_key,
        timeout=240,
        max_retries=5,
    )


async_client = AsyncOpenAI(
        base_url=xinference_api_base,
        api_key=xinference_api_key,
        timeout=240,
        max_retries=2,
    )



def eval_from_json(response: str) -> dict | None:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.lstrip("\n")
    try:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            res = ast.literal_eval(json_content)
            return res

        pattern = r"```(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            json_content = json_content.removeprefix("json")
            res = ast.literal_eval(json_content)
            return res
        
        pattern = r"```\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            res = ast.literal_eval(json_content)
            return res

        r = response.removeprefix("``\njson").removesuffix("```").strip()
        res = ast.literal_eval(r)
        return res

    except Exception as e:
        print(f"Error parsing JSON: {response} {e}")
        return None

def chat(prompt,model="gpt-4o-mini"):
    c = client
    if model.startswith("gpt-"):
        c = gpt_client

    if model =="qwen3":
        prompt = "Please do not think. \n" + prompt
        
    sys = '''You are a helpful assistant. '''
    try:
        completion = c.chat.completions.create(
            max_tokens=10000,
            model=model,
            messages=[
                {"role": "system", "content": f"{sys}"},
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        result = completion.choices[0].message
        content = result.content
        content = re.sub(r".*?</think>", "", content, flags=re.DOTALL)
        return content
    except Exception as e:
        return f"Error: {str(e)}"

async def achat(prompt,model="gpt-4o-mini"):
    c = async_client
    if model.startswith("gpt-"):
        c = gpt_async_client

    if model =="qwen3":
        prompt = "Please do not think. \n" + prompt
        
    sys = '''You are a helpful assistant. '''
    try:
        completion = await c.chat.completions.create(
            max_tokens=10000,
            model=model,
            messages=[
                {"role": "system", "content": f"{sys}"},
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        result = completion.choices[0].message
        content = result.content
        content = re.sub(r".*?</think>", "", content, flags=re.DOTALL)
        return content

        
    except Exception as e:
        return f"Error: {str(e)}"  

async def achat_and_get_json(prompt, model="gpt-4o-mini",max_tries=3):
    for i in range(max_tries):
        res = await achat(prompt, model=model)
        res = eval_from_json(res)
        if res is not None:
            return res
    return None

def chat_and_get_json(prompt, model="gpt-4o-mini",max_tries=3):
    for i in range(max_tries):
        res = chat(prompt, model=model)
        res = eval_from_json(res)
        if res is not None:
            return res
    return None
         





init(autoreset=True)

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, "")
        message = super().format(record)
        return color + message + Style.RESET_ALL

def setup_logger():
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    now.strftime("%H%M%S%f")[:-3]

    os.path.join("./logs", date_str)
    # os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()