import torch
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer, BertForSequenceClassification, BertTokenizer

# Load BlenderBot model and tokenizer
blenderbot_model = BlenderbotForConditionalGeneration.from_pretrained("C:/UK/UNI/Project/Group project/Final_Project/Models/Trained_BLENDERBOT")
tokenizer = BlenderbotTokenizer.from_pretrained("C:/UK/UNI/Project/Group project/Final_Project/Models/Trained_BLENDERBOT")

# Load the fine-tuned BERT tokenizer and model from your directory
bert_tokenizer = BertTokenizer.from_pretrained("C:/UK/UNI/Project/Group project/Final_Project/Models/Trained_BERT")
bert_model = BertForSequenceClassification.from_pretrained("C:/UK/UNI/Project/Group project/Final_Project/Models/Trained_BERT")

# Define function to generate response using BlenderBot and BERT
def get_response(conversation):
    # Encode conversation using BlenderBot tokenizer
    inputs = bert_tokenizer.encode(conversation + tokenizer.eos_token, return_tensors="pt")

    # Generate output using BlenderBot
    outputs = blenderbot_model.generate(inputs, max_length=1024, pad_token_id=tokenizer.eos_token_id)

    # Decode output using BlenderBot tokenizer
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Predict class using fine-tuned BERT model
    bert_inputs = bert_tokenizer.encode(conversation, return_tensors='pt')
    bert_outputs = bert_model(bert_inputs)[0]
    _, predicted_class = torch.max(bert_outputs, dim=1)
    predicted_class = predicted_class.item()

    return response, predicted_class
