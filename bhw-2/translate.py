import torch
from dataset import Dataset
from model import attention_mask


@torch.no_grad()
def decode(model, src, src_mask, max_length, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask).to(device)
    result = torch.full((1, 1), Dataset.BOS_IDX).long().to(device)

    for i in range(max_length - 1):
        dst_mask = attention_mask(result.shape[1]).to(device)
        out = model.decode(result, memory, dst_mask)
        logits = model.fc(out[:, -1, :])
        word = logits.argmax(dim=1).item()

        result = torch.cat([result, torch.ones(1, 1).type_as(src.data).fill_(word)], dim=1)

        if word == Dataset.EOS_IDX:
            break

    return result


@torch.no_grad()
def translate(model, src, device):
    model.eval()
    length = src.shape[0]
    src_mask = torch.zeros((length, length)).bool()
    dst_tokens = decode(model, src.unsqueeze(dim=0), src_mask, length + 5, device)
    return dst_tokens.squeeze(dim=0)
