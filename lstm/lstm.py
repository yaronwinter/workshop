import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import config as params
from utils import embedding as embedder

class LSTM(nn.Module):
    def __init__(self,
                 config: dict,
                 num_classes: int):
        # Constructor.
        super(LSTM, self).__init__()
        
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
        
        # LSTM layer.
        self.batch_first = config[params.BATCH_FIRST]
        self.hidden_dim = config[params.HIDDEN_DIM]
        self.bidirectional = bool(config[params.BIDIRECTIONAL])
        self.lstm = nn.LSTM(self.w2v_model.vectors.shape[1],
                            self.hidden_dim,
                            num_layers=config[params.NUM_LAYERS],
                            bidirectional=self.bidirectional,
                            dropout=0,
                            batch_first=self.batch_first)
        
        # Set the dropout for the embedding layer and the lstm's output layer.
        self.dropout = nn.Dropout(config[params.DROPOUT])

        # Set the last layer, fully connected.
        self.fc = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), num_classes)
        
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
        packed_text = pack_padded_sequence(embed_text, lengths, batch_first=self.batch_first)
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
        pad_packed_output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)
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
        
        return logits
    