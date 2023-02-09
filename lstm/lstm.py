# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:50:06 2020

@author: YaronWinter
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from utils import PAD_LABEL

NUM_LAYERS = 1
BIDIRECTIONAL = True

class LSTM_NLP(nn.Module):
    def __init__(self,
                 vocab,
                 num_classes,
                 hidden_dim=128,
                 dropout=0.5,
                 freeze=True,
                 batch_first=True):
        # Constructor.
        super().__init__()
        
        # Words Embedding layer.
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,
                                                      padding_idx=vocab.stoi[PAD_LABEL],
                                                      freeze=freeze)
        
        # LSTM layer.
        self.lstm = nn.LSTM(vocab.vectors.shape[1],
                            hidden_dim,
                            num_layers=NUM_LAYERS,
                            bidirectional=BIDIRECTIONAL,
                            dropout=0,
                            batch_first=batch_first)
        
        # Set the dropout for the full connected layer.
        self.dropout = nn.Dropout(dropout)
        
        # Set the last layer, fully connected.
        self.fc = nn.Linear(hidden_dim * (2 * BIDIRECTIONAL), num_classes)
        
    # The forward function.
    def forward(self, texts, lengths):
        # Get the embedding of the given text.
        # text = [#batch size, sentence length]
        #lengths = [#batch size]
        #print('text = ' + str(text.shape))
        #print('lengths = ' + str(lengths.shape))
        embed_text = self.dropout(self.embedding(texts))
        # embed_text = [#batch size, sentence length, embed dim]
        #print('embed text = ' + str(embed_text.shape))
        
        # Get the packed sentences.
        packed_text = pack_padded_sequence(embed_text, lengths, batch_first=True)
        # packed_text = [[sum lengths, embed dim], [#different lengths]]
        #print('packed text data = ' + str(packed_text.data.shape))
        #print('packed text batch sizes = ' + str(packed_text.batch_sizes.shape))
        
        # Call the LSTM layer.
        packed_output, (hidden, cell) = self.lstm(packed_text)
        # packed output = [[sum lengths, hidden dim], [#different lengths]]
        # hidden = [1, #batch size, hidden dim]
        # cell   = [1, #batch size, hidden dim]
        #print('packed out = ' + str(packed_out.data.shape))
        #print('hidden = ' + str(hidden.shape))
        #print('cell = ' + str(cell.shape))
        
        # unpack the output.
        pad_packed_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # pad_packed_output = [batch size, sentence length, hidden dim * 2]
        
        # Prmute the output before pooling.
        permuted_output = self.dropout(pad_packed_output.permute(0, 2, 1))
        # permuted_output = [batch size, hidden dim * 2, sentence length]

        # Max pooling layer.        
        pooled_output = F.max_pool1d(permuted_output, kernel_size=permuted_output.shape[2])
        # pooled_output = [batch size, hidden dim * 2, 1]
        
        # Call the linear full connected layer, after droping out.
        logits = self.fc(torch.squeeze(pooled_output, dim=2))
        # logits = [batch size, #classes]
        #print('logits = ' + str(logits.shape))
        
        
        # Perform non linearity on the final results.
        res = torch.softmax(logits, dim=1)
        
        return res
    