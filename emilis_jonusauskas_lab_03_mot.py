# -*- coding: utf-8 -*-
"""Emilis_Jonusauskas_Lab_03_mot

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Rid4pZ-MQRVO9pIokIr1naKNP_pPXqyR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import unicodedata
import requests
from bs4 import BeautifulSoup
import random

names = []
for key in ['a', 'b', 'c', 'c-2', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'r', 's', 's-2', 't', 'u', 'v', 'z', 'z-2']:
    url = f'https://vardai.vlkk.lt/sarasas/{key}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_='names_list__links names_list__links--woman')
    names += [name.text for name in links]

np.savetxt('mot_vardai.txt', names, fmt='%s', header='name', comments='', newline='\n')

"""Panaikinau kirčius, kad nesigautų vardų kaip Siontr̃ra. Taip pat panaikinau didžiąsias raides, dėl kodo efektyvumo."""

class NameDataset(Dataset):
    def __init__(self, csv_file):
        # Load and preprocess names
        self.names = self._preprocess_names(pd.read_csv(csv_file)['name'].values)

        # Build vocabulary (characters + padding space)
        lithuanian_letters = "ąčęėįšųū"
        self.chars = sorted(list(set(''.join(self.names) + lithuanian_letters + ' ')))  # Including a padding character
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

    def _preprocess_names(self, names):
        """Removes accentuation and normalizes the names."""
        return [
            ''.join(
                c for c in unicodedata.normalize('NFD', name)
                if unicodedata.category(c) != 'Mn'
            ).lower()
            for name in names
        ]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Add a padding character at the end
        name = self.names[idx] + ' '
        # Encode the name into integers
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)

dataset = NameDataset('mot_vardai.txt')

# Custom collate function for padding
def pad_collate(batch):
    padded_seqs = pad_sequence(batch, batch_first=True, padding_value=0)
    input_seq = padded_seqs[:, :-1]
    target_seq = padded_seqs[:, 1:]
    return input_seq, target_seq

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

"""Epochų skaičių palikau ties 50, nes su didesniu skaičiu vardai tampa prasti"""

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x

# Training Loop
def train_model(model, dataloader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    for epoch in range(epochs):
        model.train()  # Ensure the model is in training mode
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output.transpose(1, 2), target_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count
        print(f'Epoch {epoch+1}, Average Loss: {average_loss}')

model = MinimalTransformer(vocab_size=dataset.vocab_size, embed_size=128, num_heads=8, forward_expansion=4)
train_model(model, dataloader)

"""Įdiegiau keletą papildomų sąlygų, kad moteriški vardai būtų lietuviški :

1.    Vieni ilgiausių moteriškų vardų Lietuvoje yra sudaryti iš 10 raidžių , todėl modelis generuoja tik maksimaliai 11 raidžių vardus
2.   Vardai turi baigtis raide 'a' arba 'ė'


"""

def sample(model, dataset, start_str='a', max_length=11,temperature =1):
    assert temperature > 0
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension

        output_name = start_str
        vowels = set("aė")  # Including Lithuanian vowel 'ė'

        for _ in range(max_length - len(start_str)):
            output = model(input_seq)

            # Get the last character from the output
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)
            # Sample a character from the probability distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.int_to_char[next_char_idx]

            if next_char == ' ':  # Assume ' ' is your end-of-sequence character
                break

            output_name += next_char
            # Update the input sequence for the next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)

        # Enforce valid ending character
        if output_name[-1] not in vowels:
            output_name = output_name.rstrip()[:-1] + random.choice(list(vowels))

        return output_name

# After training your model, generate a name starting with a specific letter
for _ in range(15):
    generated_name = sample(model, dataset, start_str='skal',temperature=0.5)
    print(generated_name.capitalize())