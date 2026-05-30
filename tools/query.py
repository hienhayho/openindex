import os
import json
from dotenv import load_dotenv
from openindex import WikiQueryAgent

load_dotenv()
model_name = os.getenv("OPENAI_MODEL_NAME")
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
extra_body = json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}"))

assert all(
    [model_name, base_url, api_key]
), "Please set OPENAI_MODEL_NAME, OPENAI_BASE_URL, and OPENAI_API_KEY in your environment variables."

agent = WikiQueryAgent(
    wiki_dir="./wiki",
    model_name=model_name,
    base_url=base_url,
    api_key=api_key,
    extra_body=extra_body,
)
answer = agent.ask_sync("What is RAG ?")
print(answer)
