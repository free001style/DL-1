import torch
import torch.nn as nn
import math

from dataset import Dataset


class PositionalEncoding(nn.Module):
    def __init__(self, length, emb_dim, dropout):
        super().__init__()

        w_k = torch.exp(-torch.arange(0, emb_dim, 2) * math.log(10000) / emb_dim)

        t = torch.arange(0, length).reshape(length, 1)
        pos_embedding = torch.zeros((length, emb_dim))
        pos_embedding[:, 0::2] = torch.sin(w_k * t)
        pos_embedding[:, 1::2] = torch.cos(w_k * t)

        self.register_buffer('pos_embedding', pos_embedding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        return self.dropout(emb + self.pos_embedding[:emb.shape[1], :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_dim)


def attention_mask(size):
    mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    return mask


def create_mask(src, dst, device):
    src_len = src.shape[1]
    dst_len = dst.shape[1]

    src_mask = torch.zeros((src_len, src_len), device=device).bool()
    dst_mask = attention_mask(dst_len).to(device)

    src_padding_mask = src == Dataset.PAD_IDX
    dst_padding_mask = dst == Dataset.PAD_IDX

    return src_mask, dst_mask, src_padding_mask, dst_padding_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, dst_vocab_size, num_enc_layers,
                 num_dec_layers, feedforward_dim, d_model, num_heads, dropout, max_length=128):
        super().__init__()

        self.transformer = nn.Transformer(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=feedforward_dim,
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.d_model = d_model
        self.fc = nn.Linear(d_model, dst_vocab_size)
        self.source_emb = TokenEmbedding(vocab_size=src_vocab_size, emb_dim=d_model)
        self.target_emb = TokenEmbedding(vocab_size=dst_vocab_size, emb_dim=d_model)
        self.positional_encoding = PositionalEncoding(length=max_length, emb_dim=d_model, dropout=dropout)

        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.source_emb.embedding.weight, mean=0., std=2 / math.sqrt(self.d_model))
        nn.init.normal_(self.target_emb.embedding.weight, mean=0., std=2 / math.sqrt(self.d_model))
        nn.init.normal_(self.fc.weight, mean=0., std=2 / math.sqrt(self.d_model))
        nn.init.normal_(self.fc.bias, mean=0., std=0.)

    def forward(self, src, dst, src_mask, dst_mask, src_padding_mask, dst_padding_mask, memory_key_padding_mask):
        src_pos_enc = self.positional_encoding(self.source_emb(src))
        dst_pos_enc = self.positional_encoding(self.target_emb(dst))
        outs = self.transformer(
            src_pos_enc, dst_pos_enc, src_mask, dst_mask, None, src_padding_mask, dst_padding_mask,
            memory_key_padding_mask
        )
        return self.fc(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.source_emb(src)), src_mask)

    def decode(self, dst, memory, dst_mask):
        return self.transformer.decoder(self.positional_encoding(self.target_emb(dst)), memory, dst_mask)


def get_model(args, vocab_sizes):
    src_vocab_size, dst_vocab_size = vocab_sizes

    return Transformer(src_vocab_size=src_vocab_size, dst_vocab_size=dst_vocab_size,
                       num_enc_layers=args.num_layers,
                       num_dec_layers=args.num_layers,
                       feedforward_dim=args.feedforward_dim, d_model=args.model_dim,
                       num_heads=args.num_heads, dropout=args.dropout)
