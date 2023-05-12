import pandas as pd
import torch
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# Load the tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')

# Load the original model
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')

# Define a function to generate response
def generate_response(input_text, loaded_model=None, loaded_tokenizer=None):
    response_df = pd.read_csv('C:/UK/UNI/Project/Group project/Final_Project/Dataset/datafile.csv',header=None, names=['Questions','Answers']) 
    input_text_df = response_df['Questions'] 
    modified_response_df = response_df['Answers']  
    if input_text in input_text_df.values:  
        idx = input_text_df[input_text_df == input_text].index[0]  
        output_text = modified_response_df[idx]  
    else:
        if loaded_model and loaded_tokenizer:
            input_ids = loaded_tokenizer.encode(input_text, return_tensors='pt')
            output_ids = loaded_model.generate(input_ids)
            output_text = loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            output_ids = model.generate(input_ids)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Load the Fine-Tuned Model
loaded_model = BlenderbotForConditionalGeneration.from_pretrained('C:/UK/UNI/Project/Group project/Final_Project/Models/Trained_BLENDERBOT')

# Define a function to get chatbot responses
def get_chatbot_response(input_text):
    # Generate a response using the modified model
    response = generate_response(input_text, loaded_model, tokenizer)
    # Return the response
    return response
