import json
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausualLM,
                          BitsAndBytestConfig,
                          pipeline)
#hf account configuration
config_data = json.load(open('config.json'))
HF_TOKEN = config_data['HF_TOKEN']

model_name = "meta-llama/Meta-Llama-3-8B"

#Quantisation

bnb_config = BitsAndBytestConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

#load LLM

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)
def get_response(prompt):
  sequences = text_generator(prompt)
  gen_text = sequences[0]["generated_text"]
  return gen_text


prompt = "What is Machine Learning?"
llama3_response = get_response(prompt)
llama3_response
