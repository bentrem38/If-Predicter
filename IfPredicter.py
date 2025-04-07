from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import load_dataset
import pandas as pd
import re
from datasets import Dataset
import autopep8
import sacrebleu
import codebleu
import os
from datasets import Dataset
import torch
import evaluate
from codebleu import calc_codebleu
from tqdm import tqdm
import numpy as np
import csv

# ------------------------
# 1. Install Required Libraries
# ------------------------
#!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
#!pip install transformers datasets evaluate -q

#https://huggingface.co/Salesforce/codet5-small
# ------------------------------------------------------------------------
# 2. Load Dataset (CodeXGLUE - Code Translation Java <=> C#)
# ------------------------------------------------------------------------
data_dir = r"C:\Users\bentr\Downloads\Archive\Archive"

# CodeXGLUE is a benchmark dataset collection by Microsoft for code-related tasks.
# Here, we use the code-translation-python-java dataset.
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
# Read the CSV files into DataFrames
test_dataset = load_dataset('csv', data_files=os.path.join(data_dir, csv_files[0]))['train']
train_dataset = load_dataset('csv', data_files=os.path.join(data_dir, csv_files[1]))['train']
validation_dataset = load_dataset('csv', data_files=os.path.join(data_dir, csv_files[2]))['train']

dataset = DatasetDict({
    'test': test_dataset,
    'train': train_dataset,
    'validation': validation_dataset
})
#print(dataset)

model_checkpoint = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_tokens(["<MASK>"]) #Imagine we need an extra token. This line adds the extra token to the vocabulary

model.resize_token_embeddings(len(tokenizer))

def mask_dataset(dataset, datatype):
    if datatype == "test" or "validation": max = 4999
    if datatype == "train": max = 49999
    processed_methods = []
    processed_targets = []
    i = 0

    # Loop through the dataset and apply processing
    yes = 0
    no = 0
    while i <= max:
        # Get the current method and target block
            if (i + 1) % 250 == 0: print(f"Processed {i + 1}")
            flattened_method = dataset[datatype]["cleaned_method"][i]
            target = dataset[datatype]["target_block"][i]

        # Flatten the method by joining words with a single space
            flattened_method = " ".join(flattened_method.split())
            flattened_method = re.sub(r'\s*([=+\-*/%<>!&|^(),:{}\[\].])\s*', r'\1', flattened_method)

        # Normalize the target block
            target = re.sub(r'\s*([=+\-*/%<>!&|^(),:{}\[\].])\s*', r'\1', target)

        # Replace target with <MASK> in the flattened method
            if target not in flattened_method:
                no+=1
            if target in flattened_method:
                flattened_method = flattened_method.replace(target, "<MASK>")
                yes+=1
                processed_methods.append(flattened_method)
                processed_targets.append(target)
        # Append processed results
            i += 1
    print(yes)
    print(no)
    # Build Dataset (not DatasetDict)
    processed = Dataset.from_dict({
        'processed_target': processed_targets,
        'processed_method': processed_methods,
    })
    return processed
valid = mask_dataset(dataset, "validation")
test = mask_dataset(dataset, "test")
train = mask_dataset(dataset, "train")
#print(valid)
#print(train)
#print(test)

def preprocess_function(dataset):
    inputs = dataset["processed_method"]
    targets = dataset["processed_target"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



train = train.map(preprocess_function, batched=True)
valid = valid.map(preprocess_function, batched = True)
test = test.map(preprocess_function, batched = True)
#print(valid)
#print(train)
#print(test)

# ------------------------------------------------------------------------
# 5. Define Training Arguments and Trainer
# ------------------------------------------------------------------------


training_args = TrainingArguments(
    output_dir="./codet5-finetuned2",
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
    train_dataset=train,
    eval_dataset=valid,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ------------------------
# 6. Train the Model
# ------------------------
trainer.train()

# ------------------------
# 8. Test Code Translation
# ------------------------
model2 = model.to('cuda')
input_code = test["processed_method"][1000]
#print(test["processed_target"][1000])
inputs = tokenizer(input_code, return_tensors="pt", padding=True, truncation=True)
outputs = model2.generate(**inputs.to('cuda'), max_length=256)
print(tokenizer.decode(outputs[0]))
model2.eval()

all_inputs = test["processed_method"]
batch_size = 8  # start small, increase if your GPU can handle it
decoded_outputs = []

for i in tqdm(range(0, len(all_inputs), batch_size)):
    batch = all_inputs[i:i+batch_size]

    # Tokenize batch
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model2.generate(**inputs, max_length=256)

    # Decode each output
    decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_outputs.extend(decoded_batch)

# Optional: print a few outputs
#for i in range(5):
#    print(test["processed_target"][300+i])
#    print(f"Prediction: {decoded_outputs[300+i]}")

predictions = decoded_outputs
references = test["processed_target"]

# Evaluate overall BLEU Scores
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print("SacreBLEU Score: ", results)
res = calc_codebleu([[ref] for ref in references], predictions, lang="python")
print("Bleu Score: ", res)
"""
SacreBLEU Score:  {'bleu': 0.44042225634462967, 'precisions': [0.706239299568732, 0.5036700369434183, 0.42249273758472816, 0.3636392821599409], 'brevity_penalty': 0.9109032434424128, 'length_ratio': 0.914646474657575, 'translation_length': 46143, 'reference_length': 50449}
Bleu Score:  {'codebleu': 0.2896350403463984, 'ngram_match_score': 0.23362921904780848, 'weighted_ngram_match_score': 0.24246497036092346, 'syntax_match_score': 0.44688279301745637, 'dataflow_match_score': 0.23556317895940537}
Exact Match Score: 0.29
"""
exact_match_score = np.mean([ref == pred for ref, pred in zip(references, predictions)])
print(f"Exact Match Score: {exact_match_score:.2f}")

bleu_scores = []
exact_matches = []
codebleu_scores = []
for ref, pred in zip(references, predictions):
    # BLEU expects a list of predictions and references
    bleu_result = bleu.compute(predictions=[pred], references=[ref])
    bleu_scores.append(bleu_result["bleu"]*100)


    # CodeBLEU - assumes `calc_codebleu` can be called per-sample
    codebleu_result = calc_codebleu([[ref]], [pred], lang="python")
    codebleu_scores.append(codebleu_result["codebleu"]*100)

    results = {
    "Input function with masked if condition": test["processed_method"],
    "Was the prediction correct (exact match)?": [ref == pred for ref, pred in zip(references, predictions)],
    "Expected if condition" : test["processed_target"],
    "Predicted if condition": predictions,
    "CodeBLEU prediction score": codebleu_scores, 
    "BLEU-4 prediction score": bleu_scores
}
df = pd.DataFrame(results)
df.to_csv("testset-results.csv", index=False)