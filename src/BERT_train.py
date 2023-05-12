# !pip install transformers
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
import numpy as np
import pandas as pd

device = torch.device('cuda')
print(device)

seed = 97
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Load tokenizer and pre-trained BERT model (Bert-base-uncased)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# num_labels to 5  [0- Happy, 1-Joy, 2- Neutral, 3-Sad, 4-Fear]
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5) 
model.to(device)

#Read training data from CSV file (try pasting the path, if this ain't working)
train_data = pd.read_csv('batch1_test.csv', header=None, names=['column 1', 'column 2'])

# Shuffling it as the labeled dataset ('batch1_test.csv') is not randomized - Preventing unequal learning of the model
shuffled_set = train_data.sample(frac=1, random_state=42)

# Extracting the data from the shuffled set
train_texts = shuffled_set['column 1'].tolist()
train_labels = shuffled_set['column 2'].tolist()

# Tokenize and encode training data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# train_dataset = list(zip(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels))
train_dataset = list(zip(torch.tensor(train_encodings['input_ids']).to(device), torch.tensor(train_encodings['attention_mask']).to(device), torch.tensor(train_labels).to(device)))

# Optimize with learning rate 
optimizer = AdamW(model.parameters(), lr=5e-7)

# Training the model for 60 epochs
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    print(epoch)
    for input_ids, attention_mask, labels in train_dataset:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.unsqueeze(0).to(device), attention_mask=attention_mask.unsqueeze(0).to(device), labels=torch.tensor([labels]).to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()

try:
    model.save_pretrained('Zen_Model_B1')
except Exception as e:
    print(e)