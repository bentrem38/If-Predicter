# Hugging Face - Fine-Tuning CodeT5 for Code Translation (AI4SE Focus)

# This notebook demonstrates how to fine-tune the CodeT5 model using Hugging Face Transformers
# for a Software Engineering task: translating Python code to Java.

# ------------------------
# 1. Install Required Libraries
# ------------------------
#!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#!pip install transformers datasets evaluate -q

# ------------------------------------------------------------------------
# 2. Load Dataset (CodeXGLUE - Code Translation Java <=> C#)
# ------------------------------------------------------------------------
from datasets import load_dataset


# CodeXGLUE is a benchmark dataset collection by Microsoft for code-related tasks.
# Here, we use the code-translation-python-java dataset.
dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")

# Dataset contains: 'train', 'validation', 'test' splits
print("Sample Python Code:", dataset['train'][0]['java'])
print("Target Java Code:", dataset['train'][0]['cs'])

# ------------------------------------------------------------------------
# 3. Load Pre-trained Model & Tokenizer
# ------------------------------------------------------------------------
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

model_checkpoint = "Salesforce/codet5-small"

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<IF-STMT>"]) #Imagine we need an extra token. This line adds the extra token to the vocabulary

model.resize_token_embeddings(len(tokenizer))


# ------------------------------------------------------------------------------------------------
# 4. We prepare now the fine-tuning dataset using the tokenizer we preloaded
# ------------------------------------------------------------------------------------------------

def preprocess_function(examples):
    inputs = examples["java"]
    targets = examples["cs"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ------------------------------------------------------------------------
# 5. Define Training Arguments and Trainer
# ------------------------------------------------------------------------


training_args = TrainingArguments(
    output_dir="./codet5-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ------------------------
# 6. Train the Model
# ------------------------
trainer.train()

# ------------------------
# 7. Evaluate on Test Set
# ------------------------
metrics = trainer.evaluate(tokenized_datasets["test"])
print("Test Evaluation Metrics:", metrics)

# ------------------------
# 8. Test Code Translation
# ------------------------
input_code = "def add(a, b):\n    return a + b"
inputs = tokenizer(input_code, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_length=256)
print("Generated Java Code:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))