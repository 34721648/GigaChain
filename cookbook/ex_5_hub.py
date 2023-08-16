from langchain.prompts import load_prompt

prompt = load_prompt("lc://prompts/hello-world/prompt.yaml")
print("OK")