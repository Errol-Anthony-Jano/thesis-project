# beta version: no feedback loop yet
import json
from generation_pipeline_new import run_generation_pipeline

list = []

# start the llm
with open("./datasets/ULT.jsonl", "r") as file:
    data = json.load(file)
    list = data[:10]

run_generation_pipeline(list)