# beta version: no feedback loop yet
import json
from generation_pipeline import run_generation_pipeline

list = []

with open("./datasets/ULT_Lite.jsonl", "r") as file:
    data = json.load(file)
    list = data[:10]

run_generation_pipeline(list)