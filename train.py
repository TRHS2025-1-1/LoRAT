import math
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
from LoRAT import LoRAT

model_path = 'bert_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model=BertModel.from_pretrained(model_path)
model = LoRAT(model)

example_text = "This is a test sentence."
inputs = tokenizer(example_text, return_tensors='pt')

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']


with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)


print(outputs)
