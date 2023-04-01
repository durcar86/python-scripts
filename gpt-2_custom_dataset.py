## Complex code for fine-tuning GPT-2 on a custom dataset using the Hugging Face Transformers library
#
#

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm

# Define the dataset
class MyDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = f.readlines()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index].strip()
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor(input_ids[:-1]), torch.tensor(input_ids[1:])

# Define the model and optimizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define the training loop
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for input_ids, labels in tqdm(dataloader, desc="Training"):
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

# Set up the training data and dataloader
train_data = MyDataset("my_dataset.txt")
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)

# Fine-tune the GPT-2 model
num_epochs = 3
for epoch in range(num_epochs):
    loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Epoch {epoch+1} loss: {loss:.2f}")

# Save the fine-tuned model
model.save_pretrained("my_model")
