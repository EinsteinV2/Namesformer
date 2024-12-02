import streamlit as st
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

import sys
sys.stdout.reconfigure(encoding='utf-8')

class NameDataset(Dataset):
    def __init__(self, csv_file):
        # Load and preprocess names
        self.names = self._preprocess_names(pd.read_csv(csv_file)['name'].values)

        # Build vocabulary (characters + padding space)
        lithuanian_letters = "ąčęėįšųū"
        self.chars = sorted(list(set(''.join(self.names)+ lithuanian_letters + ' ')))  # Including a padding character
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

model_path_vyr = r"namesformer_model_male.pt"
model_vyr = torch.load(model_path_vyr)
model_vyr.eval()

model_path_mot = r"namesformer_model_female.pt"
model_mot = torch.load(model_path_mot)
model_mot.eval()


dataset_vyr = NameDataset(r"vardai_vyr.txt")


dataset_mot = NameDataset(r"vardai_mot.txt")


def sample_vyr(model_vyr, dataset, start_str='a', max_length=13,temperature=0.001):
    assert temperature > 0
    model_vyr.eval()  # Switch to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension

        output_name = start_str
        vowels = set("aeiouy")  # Allowed vowels

        for _ in range(max_length - len(start_str)):
            output = model_vyr(input_seq)

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

        # Enforce the name ends with a vowel followed by 's'
        if len(output_name) < max_length:
            # If it doesn't already end with a vowel and 's', adjust it
            if len(output_name) < 2 or output_name[-1] != 's' or output_name[-2] not in vowels:
                last_vowel = random.choice(list(vowels))  # Randomly pick a vowel
                output_name = output_name.rstrip()[:-1] + last_vowel + 's'

        return output_name
    
def sample_mot(model_mot, dataset, start_str='a', max_length=11,temperature=0.001):
    assert temperature > 0
    model_mot.eval()  # Switch to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension

        output_name = start_str
        vowels = set("aė")  # Including Lithuanian vowel 'ė'

        for _ in range(max_length - len(start_str)):
            output = model_mot(input_seq)

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

# Streamlit UI
st.title("Lithuanian Name Generator")
st.write("Generate Lithuanian male or female names starting with specific letters.")

# User Inputs
prefix = st.text_input("Enter up to 5 starting LOWERCASE letters:", max_chars=5)
gender = st.radio("Select Gender:", ("Male", "Female"))
temperature = st.slider("Set Creativity:", min_value=0.001, max_value=1.5, value=1.0, step=0.01)

# Generate Name Button
if st.button("Generate Name"):
    if not prefix:
        st.error("Please enter a prefix.")
    elif len(prefix) > 5:
        st.error("Prefix cannot be longer than 5 characters.")
    else:
        try:
            # Call the appropriate sample function based on gender
            if gender == "Male":
                name = sample_vyr(model_vyr,dataset_vyr, start_str=prefix, max_length=13, temperature=temperature)
                Name = name.capitalize()
            else:
                name = sample_mot(model_mot, dataset_mot, start_str=prefix, max_length=11, temperature=temperature)
                Name = name.capitalize()
            
            st.success(f"Generated Name: {Name}")
        except Exception as e:
            st.error(f"An error occurred while generating the name: {e}")
