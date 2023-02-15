import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.embedding as embedder
from utils import config as params
import numpy as np

class CNN(nn.Module):
    def __init__(self,
                 config: dict,
                 num_classes):
        super(CNN, self).__init__()

        print('Freeze embedding matrix = ' + str(config[params.FREEZE_EMBEDDING]))
        print('load embedding model')
        added_words = [params.PAD_LABEL, params.UNK_LABEL]
        self.w2v_model = embedder.Embedded_Words(config[params.EMBED_WORDS_FILE], added_words, config[params.NORM_EMBED_VECS])
        print('\tw2v after padding: ' + str(self.w2v_model.vectors.shape))

        print('generate embedding tensor')
        # Set the embedding module.
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.w2v_model.vectors),
            padding_idx = self.w2v_model.w2i[params.PAD_LABEL],
            freeze=config[params.FREEZE_EMBEDDING])
        
        print('allocate convolution  layers')
        filter_width = config[params.CNN_OUT_CHANNELS]
        kernels = config[params.CNN_KERNELS]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.w2v_model.vectors.shape[1],
                      out_channels=filter_width,
                      kernel_size=kernel)
            for kernel in kernels
        ])

        self.fc = nn.Linear(filter_width * len(kernels), num_classes)
        self.dropout = nn.Dropout(p=config[params.DROPOUT])

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
