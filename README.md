# ThesisProject (title otw)

## Prerequisites
1. Python 3.12 >= 
2. Ollama installed

## Usage Instructions

### Ollama Installation and Setup
1. Install Ollama from https://ollama.com/download; follow the instructions.
2. From your terminal, execute the following command:

```ollama run qwen2.5:3b```

This will automatically download the model for you. After installing, it will run automatically.


### Using the Repository
1. Clone this repository locally.
2. Create a virtual environment via ```python -m venv .venv```. Note that you can name your venv anything; ```.venv ``` is just an example.
3. Activate the environment. 
* Linux/macOS: ```source venv/bin/activate```
* Windows: ```venv\Scripts\activate```
4. Install the requirements via ```pip install -r requirements.txt```. 
5. Run the program through `main.py`:

```powershell
python main.py --dataset ULT
```

Choose `ULT`, `ULT_Lite`, or `leetcode-py-all` with `--dataset`. The first two
datasets use the existing generation pipeline; `leetcode-py-all` uses a
separate pipeline that reads its `python_solution` and `description` fields.
By default, the first 10 records are processed. Set `--limit 0` to process the
entire dataset, for example:

```powershell
python main.py --dataset leetcode-py-all --limit 0
```

LeetCode test files are saved in `generated_tests/` with their task number in
the filename so records sharing a function name do not overwrite each other.

### Test Generation
Each function is sent through a multi-turn chat by default: the first turn
requests one pytest test, and each later turn includes the earlier conversation
and asks for a distinct test. The number of tests per function is controlled by
`NUM_GENERATION_ROUNDS` in `llm_config.py`. Generated test code is stored in
order in the task records returned by the pipeline and combined in each
function's output file.