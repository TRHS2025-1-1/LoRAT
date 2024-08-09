# LoRAT: Low-Rank Adaptation for Transformers

## Overview

`LoRAT` (Low-Rank Adaptation for Transformers) is a PyTorch module designed to integrate low-rank adaptation into transformer-based models, such as BERT. LoRAT enables efficient fine-tuning of large models by incorporating low-rank matrices into specific layers, thus reducing the number of trainable parameters and computational overhead.

## Features

- **Low-Rank Adaptation**: Integrates low-rank matrices into transformer layers, reducing computational and memory costs.
- **Selective Tuning**: Allows tuning specific parts of the model, such as query, key, value, or output dense layers.
- **Efficiency**: Optimizes a subset of parameters while keeping the majority of the model parameters fixed.

## Installation

Ensure you have PyTorch and Hugging Face's Transformers library installed. You can install them via pip:

```bash
pip install torch transformers
```

## Usage

Here is a step-by-step guide to using the `LoRAT` class with a pre-trained BERT model.

### 1. Import Required Libraries

```python
import math
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from LoRAT import LoRAT
```

### 2. Load Pre-trained Model and Tokenizer

Load a pre-trained BERT model and tokenizer:

```python
model_path = 'bert_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
```

### 3. Integrate LoRAT

Wrap the pre-trained model with the `LoRAT` class:

```python
model = LoRAT(model)
```

### 4. Tokenize Input Text and Perform Inference

Tokenize the input text and use the model for inference:

```python
example_text = "This is a test sentence."
inputs = tokenizer(example_text, return_tensors='pt')

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Print the output
print(outputs)
```

### 5. Model Training (Optional)

To fine-tune the `LoRAT` model, you can use the Hugging Face Trainer API. This step assumes you have a dataset and training configuration:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## LoRAT Class Explanation

The `LoRAT` class modifies a transformer model to incorporate low-rank adaptation. The key components include:

- **`rank`**: The rank of the low-rank matrices.
- **`alpha`**: A scaling factor for the low-rank matrices.
- **`tune`**: Specifies which parts of the model to tune (e.g., query, key, value, or output layers).
- **`mode`**: Determines the tuning mode (e.g., 'all' for tuning all layers).

For detailed implementation, refer to the `LoRAT` class in the `LoRAT.py` file.

## Acknowledgments

This code is based on the [LoRA](https://github.com/microsoft/LoRA) implementation provided by Microsoft. We extend our gratitude to the original authors for their work, which served as the foundation for this adaptation.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Contact

For any questions or feedback, please contact [huangswt@stu.xju.edu.cn](mailto:huangswt@stu.xju.edu.cn).
