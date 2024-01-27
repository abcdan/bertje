from transformers import BertForMaskedLM, BertTokenizer, pipeline

# Load the trained BERT model
model = BertForMaskedLM.from_pretrained("./bert_custom_trained_model")
tokenizer = BertTokenizer.from_pretrained("./custom_tokenizer")

# Set up the pipeline for text generation
fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

# Create a chat-like interface
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chat ended.")
        break
    else:
        # Generate a response based on the user input
        response = fill_mask(user_input)
        print("Model:", response[0]['sequence'])