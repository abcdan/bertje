import os
from transformers import BertConfig, BertForMaskedLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tokenizers import BertWordPieceTokenizer

base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

data_dir = os.path.join(base_dir, "data")
data_files = [os.path.join(root, name)
              for root, dirs, files in os.walk(data_dir)
              for name in files
              if name.endswith((".txt"))]

custom_dataset = []
for file in data_files:
    with open(file, 'r', encoding='utf-8') as f:
        custom_dataset.append(f.read())

tokenizer = BertWordPieceTokenizer()
tokenizer.train_from_iterator(custom_dataset, vocab_size=30000, min_frequency=2)
tokenizer.save_model(os.path.join(base_dir, "model"))

config = BertConfig(vocab_size=30000)
model = BertForMaskedLM(config)

text_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=data_dir, 
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

def check_and_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

output_dir = os.path.join(base_dir, "bert_custom_model")
check_and_create_dir(output_dir)

logging_dir = os.path.join(base_dir, "logs")
check_and_create_dir(logging_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    warmup_steps=500,
    logging_dir=logging_dir,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=text_dataset,
)

trainer.train()

model_dir = os.path.join(base_dir, "bert_custom_trained_model")
check_and_create_dir(model_dir)
model.save_pretrained(model_dir)
