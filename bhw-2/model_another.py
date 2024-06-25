import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, is_decoder=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.is_decoder = is_decoder

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_kv = nn.Linear(d_model, 2 * d_model)
        self.output_fc = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query_seq, key_val_seq, key_val_lengths):
        """
        :param query_seq: batch_size x L_query_seq x d_model
        :param key_val_seq: batch_size x L_key_val_seq x d_model
        :param key_val_lengths: batch_size (true length of key_value seq)
        :return: batch_size x L_query_seq x d_model
        """
        d_query = d_key = d_value = self.d_model // self.num_heads
        batch_size = query_seq.shape[0]
        self_attention = torch.equal(query_seq, key_val_seq)
        skip_connection = query_seq.clone()
        query_seq = self.layer_norm(query_seq)
        if self_attention:
            key_val_seq = self.layer_norm(key_val_seq)
        query = self.fc_q(query_seq)  # batch_size x L_query_seq x head * d_query
        key, value = self.fc_kv(
            key_val_seq).split(split_size=self.d_model,
                               dim=-1)  # batch_size x L_key_val_seq x head * d_key, batch_size x L_key_val_seq x head * d_value

        query = query.contiguous().view(batch_size, -1, self.num_heads,
                                        d_query)  # batch_size x L_query_seq x head x d_query
        key = key.contiguous().view(batch_size, -1, self.num_heads, d_key)  # batch_size x L_key_val_seq x head x d_key
        value = value.contiguous().view(batch_size, -1, self.num_heads,
                                        d_value)  # batch_size x L_key_val_seq x head x d_value
        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1,
                                                            d_query)  # batch_size*head x L_query_seq x d_query
        key = key.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1,
                                                        d_key)  # batch_size*head x L_key_val_seq x d_key
        value = value.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1,
                                                            d_value)  # batch_size*head x L_key_val_seq x d_value

        attention = torch.bmm(query, key.permute(0, 2, 1)) * (
                1 / math.sqrt(d_query))  # batch_size*head x L_query_seq x L_key_val_seq

        not_pad_in_keys = torch.LongTensor(range(value.shape[1])).unsqueeze(0).unsqueeze(0).expand_as(
            attention).to(device)  # batch_size*head x L_query_seq x L_key_val_seq
        not_pad_in_keys = not_pad_in_keys < key_val_lengths.repeat_interleave(self.num_heads).unsqueeze(
            1).unsqueeze(2).expand_as(attention)  # batch_size*head x L_query_seq x L_key_val_seq
        attention = attention.masked_fill(~not_pad_in_keys, -math.inf)  # batch_size*head x L_query_seq x L_key_val_seq

        if self.is_decoder and self_attention:
            value_mask = torch.tril(torch.ones_like(attention)).bool().to(
                device)  # batch_size*head x L_query_seq x L_key_val_seq
            attention = attention.masked_fill(~value_mask, -math.inf)  # batch_size*head x L_query_seq x L_key_val_seq

        attention = F.softmax(attention, dim=-1)  # batch_size*head x L_query_seq x L_key_val_seq
        attention = self.dropout(attention)

        attention = torch.bmm(attention, value)  # batch_size*head x L_query_seq x d_value
        attention = attention.contiguous().view(batch_size, self.num_heads, -1, d_value).permute(0, 2, 1,
                                                                                                 3)  # batch_size x L_query_seq x head x d_value
        attention = attention.contiguous().view(batch_size, -1, self.d_model)  # batch_size x L_query_seq x d_model

        return self.dropout(self.output_fc(attention)) + skip_connection


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_length=256):
        super(PositionEncoding, self).__init__()
        self.pos_encoding = torch.zeros((max_length, d_model))
        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)).unsqueeze(0)

        arguments = positions * freq

        self.pos_encoding[:, 0::2] = torch.sin(arguments)
        self.pos_encoding[:, 1::2] = torch.cos(arguments)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)  # 1 x max_length x d_model
        self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self):
        return self.dropout(self.pos_encoding)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc_1 = nn.Linear(d_model, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, d_model)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.GELU()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, seq):
        skip_connection = seq.clone()
        seq = self.layer_norm(seq)
        return self.dropout(self.fc_2(self.dropout(self.relu(self.fc_1(seq))))) + skip_connection


class Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, d_hidden, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionEncoding(d_model, dropout)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = d_hidden
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.MultiHeadAttention = MultiHeadAttention(d_model, num_heads, dropout)
        self.FeedForward = FeedForward(d_model, d_hidden, dropout)

    def forward(self, source, source_lengths):
        """
        :param source: batch_size x L_seq
        :param source_lengths: batch_size
        :return:
        """
        embedding = self.embedding(source)  # batch_size x L_seq x d_model
        embedding = embedding * math.sqrt(self.d_model) + self.position_embedding()[:, :embedding.shape[1]].to(device)
        embedding = self.dropout(embedding)
        for i in range(self.num_layers):
            embedding = self.MultiHeadAttention(embedding, embedding, source_lengths)
            embedding = self.FeedForward(embedding)

        return self.layer_norm(embedding)


class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, d_hidden, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionEncoding(d_model, dropout)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = d_hidden
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.SelfMultiHeadAttention = MultiHeadAttention(d_model, num_heads, dropout, True)
        self.CrossMultiHeadAttention = MultiHeadAttention(d_model, num_heads, dropout, True)
        self.FeedForward = FeedForward(d_model, d_hidden, dropout)

    def forward(self, target, target_lengths, encoder_output, encoder_output_lengths):
        """
        :param target: batch_size x L_target_seq
        :param target_lengths: batch_size
        :param encoder_output: batch_size x L_source_seq x d_model
        :param encoder_output_lengths: batch_size
        :return:
        """
        embedding = self.embedding(target)  # batch_size x L_seq x d_model
        embedding = embedding * math.sqrt(self.d_model) + self.position_embedding()[:, :embedding.shape[1]].to(device)
        embedding = self.dropout(embedding)
        for i in range(self.num_layers):
            embedding = self.SelfMultiHeadAttention(embedding, embedding, target_lengths)
            embedding = self.CrossMultiHeadAttention(embedding, encoder_output, encoder_output_lengths)
            embedding = self.FeedForward(embedding)

        return self.layer_norm(embedding)


class Transformer(nn.Module):
    def __init__(self, d_model, encoder_vocab_size, decoder_vocab_size, num_layers, num_heads, d_hidden, dropout):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(d_model, encoder_vocab_size, num_layers, num_heads, d_hidden, dropout)
        self.Decoder = Decoder(d_model, decoder_vocab_size, num_layers, num_heads, d_hidden, dropout)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, decoder_vocab_size)

    def _weight_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.)

        nn.init.normal_(self.Encoder.embedding.weight, mean=0., std=2 / math.sqrt(self.d_model))
        nn.init.normal_(self.Decoder.embedding.weight, mean=0., std=2 / math.sqrt(self.d_model))
        nn.init.normal_(self.fc.weight, mean=0., std=2 / math.sqrt(self.d_model))
        nn.init.normal_(self.fc.bias, mean=0., std=0)

    def forward(self, source_sequence, target_sequence, source_length, target_length):
        encoder_seq = self.Encoder(source_sequence, source_length)
        decoder_seq = self.Decoder(target_sequence, target_length, encoder_seq, source_length)
        return self.fc(decoder_seq)


def make_model(config, encoder_vocab_size, decoder_vocab_size):
    return Transformer(config['model']['d_model'], encoder_vocab_size, decoder_vocab_size,
                       config['model']['num_layers'],
                       config['model']['num_heads'], config['model']['d_hidden'], config['model']['dropout'])
