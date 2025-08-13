import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from openai import OpenAI

# Resolve projeto e .env
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"

print(f"[DEBUG] Looking for .env at: {env_path}")
print(f"[DEBUG] .env exists: {env_path.exists()}")

# 1) Tenta carregar normalmente
loaded = False
if env_path.exists():
    # encoding ajuda em casos com BOM/acentos
    try:
        loaded = load_dotenv(env_path, override=False, encoding="utf-8")
        print(f"[DEBUG] load_dotenv returned: {loaded}")
    except TypeError:
        # versões antigas não suportam 'encoding'
        loaded = load_dotenv(env_path, override=False)
        print(f"[DEBUG] load_dotenv (no encoding) returned: {loaded}")
else:
    # fallback para procurar automaticamente
    loaded = load_dotenv()
    print(f"[DEBUG] load_dotenv (auto) returned: {loaded}")

# 2) Fallback manual: se ainda não carregou OPENAI_API_KEY, lê o arquivo e injeta
if not os.getenv("OPENAI_API_KEY") and env_path.exists():
    kv = dotenv_values(env_path)
    print(f"[DEBUG] dotenv_values keys: {list(kv.keys())}")
    for k, v in kv.items():
        if v is None:
            continue
        # strip para remover espaços/aspas acidentais
        val = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, val)

api_key = os.getenv("OPENAI_API_KEY")
print(f"[DEBUG] OPENAI_API_KEY loaded: {'Yes' if api_key else 'No'}")
print(f"[DEBUG] CWD at import time: {os.getcwd()}")

if not api_key:
    raise RuntimeError(
        f"OPENAI_API_KEY not found. Checked .env at: {env_path}\n"
        f"Ensure the file contains a line like: OPENAI_API_KEY=sk-...\n"
        f"Tip: In PowerShell, test with: (Get-Content {env_path}). For CMD: type {env_path}"
    )

_client = OpenAI(api_key=api_key)


def chat_complete(
    messages_or_text,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")

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
    return resp.choices[0].message.content or ""


def chat(system_prompt: str, user_prompt: str, **kwargs) -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return chat_complete(msgs, **kwargs)