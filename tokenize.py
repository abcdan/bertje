from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

# Get the absolute path to the data directory
data_dir = Path(__file__).resolve().parent / "data"
file_paths = [str(file) for file in data_dir.glob("*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=file_paths, vocab_size=52000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("bertje")