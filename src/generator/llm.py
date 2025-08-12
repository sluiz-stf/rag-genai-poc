import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # carrega .env
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_complete(messages_or_text, model: str | None = None, temperature: float = 0.2, max_tokens: int = 600) -> str:
    model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    # Adapta se vier string
    if isinstance(messages_or_text, str):
        messages = [{"role": "user", "content": messages_or_text}]
    else:
        messages = messages_or_text
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# opcional: helper para uso simples
def chat(system_prompt: str, user_prompt: str, **kwargs) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return chat_complete(msgs, **kwargs)