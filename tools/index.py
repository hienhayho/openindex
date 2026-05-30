import os
import json
from dotenv import load_dotenv
from openindex import PageIndex, TreeConfig

load_dotenv()
model_name = os.getenv("OPENAI_MODEL_NAME")
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
extra_body = json.loads(os.getenv("OPENAI_EXTRA_BODY", "{}"))

assert all(
    [model_name, base_url, api_key]
), "Please set OPENAI_MODEL_NAME, OPENAI_BASE_URL, and OPENAI_API_KEY in your environment variables."

index = PageIndex(
    model_name=model_name,
    base_url=base_url,
    api_key=api_key,
    extra_body=extra_body,
    config=TreeConfig(max_parallel_llm_calls=8),
)
result = index.build_wiki_sync("samples/2312.10997v5.pdf", "./wiki")

index.print_result(result=result)
