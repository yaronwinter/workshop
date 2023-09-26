import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import config as params
from utils import embedding as embedder

BATCH_FIRST = True

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
        hidden_dim = config[params.HIDDEN_DIM]
        bidirectional = bool(config[params.BIDIRECTIONAL])
        self.lstm = nn.LSTM(self.w2v_model.vectors.shape[1],
                            hidden_dim,
                            num_layers=config[params.NUM_LAYERS],
                            bidirectional=bidirectional,
                            dropout=0,
                            batch_first=BATCH_FIRST)
        
        # Set the dropout for the embedding layer and the lstm's output layer.
        self.dropout = nn.Dropout(config[params.DROPOUT])

        # Set the last layer, fully connected.
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def embed_text(self, texts: torch.Tensor):
        return self.dropout(self.embedding(texts))
        
    # The forward function.
    def forward(self, embedded_text: torch.Tensor, lengths: torch.Tensor):
        # Get the embedding of the given text.
        # embedded_text = [#batch size, sentence length, embed dim]
        #lengths = [#batch size]
        
        # Get the packed sentences.
        packed_text = pack_padded_sequence(embedded_text, lengths, batch_first=BATCH_FIRST)
        # packed_text = [[sum lengths, embed dim], [#sequence length (#active batch items for each length)]]
        
        # Call the LSTM layer.
        packed_output, (hidden, cell) = self.lstm(packed_text)
        # packed output = [[sum lengths, hidden dim], [#sequence length]]
        # hidden = [1 (2 if bidirectional), #batch size, hidden dim]
        # cell   = [1 (2 if bidirectional), #batch size, hidden dim]
        
        # unpack the output.
        pad_packed_output, _ = pad_packed_sequence(packed_output, batch_first=BATCH_FIRST)
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
        
        return logits
