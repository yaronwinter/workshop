# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:55:33 2020

@author: YaronWinter
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import embedding
import utils
import numpy as np

class CNN_NLP(nn.Module):
    def __init__(self,
                 w2v_file,
                 num_classes,
                 windows=[1, 2, 3, 5],
                 width=50,
                 dropout=0.5,
                 freeze_embedding=True):
        super(CNN_NLP, self).__init__()

        print('Freeze embedding matrix = ' + str(freeze_embedding))
        print('load embedding model')
        self.w2v_model = embedding.load_embedding_model(w2v_file)
        print('\tw2v original: ' + str(self.w2v_model.vectors.shape))
        added_words = [utils.PAD_LABEL, utils.UNK_LABEL]
        added_vecs = [np.zeros(self.w2v_model.vector_size) for i in range(len(added_words))]
        self.w2v_model.add_vectors(added_words, added_vecs)
        print('\tw2v after padding: ' + str(self.w2v_model.vectors.shape))

        print('generate embedding tensor')
        # With pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.w2v_model.vectors),
            padding_idx = self.w2v_model.key_to_index[utils.PAD_LABEL],
            freeze=freeze_embedding)
        
        print('allocate convolution  layers')
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.w2v_model.vectors.shape[1],
                      out_channels=width,
                      kernel_size=windows[i])
            for i in range(len(windows))
        ])

        self.fc = nn.Linear(width * len(windows), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        # input_ids = [#batch size, sentence len]
        x_embed = self.embedding(input_ids).float()
        #x_embed = [batch size, sentence len, embed dim]
        
        x_reshape = x_embed.permute(0, 2, 1)
        # x_reshape = [batch size, embed dim, sentence len]
        
        x_conv_list = [F.relu(conv(x_reshape)) for conv in self.convs]
        # x_conv = [batch size, out_channel_width, sentence len - kernel size + 1]
        
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
        # x_pool = [batch size, out_channel_width, 1]
        
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        # x_fc = [batch size, out_channel_width * #filters]
        
        logits = self.fc(self.dropout(x_fc))
        # logits = [batch size, #classes]
        
        return logits
    
    def get_w2v_model(self):
        return self.w2v_model
