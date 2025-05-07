# myGPT
A GPT-esque UI using free and publicly available LLMs stored locally. 
No internet required and all data stays on your machine.

## Supported LLM backends: 
[x] Ollama

## Installation
```
pip install -r requirements.txt
```
### Dependencies
You need [Ollama](https://ollama.com/download) downloaded and installed on your machine first. Goto download page directly: https://ollama.com/download

## Run
```
python app.py 
```

### Note: 
1. To run and create a live link, 
```
python app.py --share
```
If run successfully, you will see a link of the form: https://48e6eb31eexxxxx.gradio.live. which can be shared with others to use. But you need to have your computer (server) be connected to internet (which acts as a server).


2. You can set the LLM model name in .env file 
OLLAMA_MODEL=<YOUR_LOCALLY_AVAILABLE_LLM_MODEL_NAME> 

Example:
```
OLLAMA_MODEL=gemma3:1b
```

3. For a list of available LLM models on Ollama, goto [models](https://ollama.com/library)


# TODO
- [ ] Add LLM model switch menu
- [ ] Add llama.cpp backend support

