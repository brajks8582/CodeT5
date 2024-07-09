# fine_tune_codet5.py

import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Load your dataset
df = pd.read_csv("code_reviews.csv")  # Assume this CSV has 'code' and 'review' columns
dataset = Dataset.from_pandas(df)

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = ["review: " + code for code in examples["code"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(text_target=examples["review"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./finetuned_codet5")
tokenizer.save_pretrained("./finetuned_codet5")
