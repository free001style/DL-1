import os
from tqdm import tqdm
import torch
import torch.nn as nn

from dataset import Dataset
from model import get_model, create_mask, translate
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class Trainer:
    def __init__(self, args, vocab_sizes, vocab, exp_name):
        self.model = get_model(args, vocab_sizes)
        self.device = device
        self.model = self.model.to(self.device)

        self.optimizer = self.get_optimizer(self.model, args)
        self.criterion = nn.CrossEntropyLoss(ignore_index=Dataset.PAD_IDX, label_smoothing=args.label_smoothing)

        self.cur_epoch = 0
        self.epochs = args.epochs
        self.checkpoint_dir = args.checkpoint_dir
        self.vocab = vocab
        self.path_for_val = os.path.join(args.datadir, args.val_target)

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.exp_name = exp_name
        run_dir = os.path.join(self.checkpoint_dir, exp_name)

        if not os.path.exists(run_dir):
            os.mkdir(run_dir)

    @staticmethod
    def get_optimizer(model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps)

    def train_epoch(self, loader, tqdm_desc):
        self.model.train()
        losses = 0

        for src, _, dst, _ in tqdm(loader, desc=tqdm_desc):
            self.optimizer.zero_grad()

            src = src.to(self.device)
            dst = dst.to(self.device)

            dst_input = dst[:, :-1]

            src_mask, dst_mask, src_padding_mask, dst_padding_mask = create_mask(src, dst_input, self.device)
            logits = self.model(src, dst_input, src_mask, dst_mask, src_padding_mask, dst_padding_mask,
                                src_padding_mask)

            dst_out = dst[:, 1:]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), dst_out.reshape(-1))

            loss.backward()
            self.optimizer.step()

            losses += loss.item() * src.shape[0]

        return losses / len(loader.dataset)

    @torch.no_grad()
    def validate_epoch(self, loader, tqdm_desc):
        self.model.eval()
        losses = 0

        for src, _, dst, _ in tqdm(loader, desc=tqdm_desc):
            src = src.to(self.device)
            dst = dst.to(self.device)

            dst_input = dst[:, :-1]

            src_mask, dst_mask, src_padding_mask, dst_padding_mask = create_mask(src, dst_input, self.device)
            logits = self.model(src, dst_input, src_mask, dst_mask, src_padding_mask, dst_padding_mask,
                                src_padding_mask)

            dst_out = dst[:, 1:]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), dst_out.reshape(-1))

            loss.backward()
            self.optimizer.step()

            losses += loss.item() * src.shape[0]

        return losses / len(loader.dataset)

    @torch.no_grad()
    def calculate_bleu(self, val_loader, tqdm_desc):
        translated = []
        for src, lengths, _, _ in tqdm(val_loader, desc=tqdm_desc):
            for i in range(src.shape[0]):
                dst_tokens = list(translate(self.model, src[i, :lengths[i].item()], self.device).cpu().numpy())
                sentence = " ".join(self.vocab.lookup_tokens(dst_tokens)).replace("<bos>", "").replace("<eos>",
                                                                                                       "").strip()
                translated.append(sentence)
        with open('val_predict.txt', "w", encoding='utf-8') as f:
            for sentence in translated:
                f.write(sentence + "\n")
        bleu = os.popen(f'cat val_predict.txt | sacrebleu {self.path_for_val}  --tokenize none --width 2 -b').read()
        os.system('rm val_predict.txt')
        return bleu

    def load_from_checkpoint(self, epoch):
        path = self.get_checkpoint_name(epoch)
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.cur_epoch = checkpoint["epoch"]

    def get_checkpoint_name(self, epoch):
        return os.path.join(self.checkpoint_dir, self.exp_name, "epoch_{}.ckpt".format(epoch))

    def save_checkpoint(self, epoch, val_loss):
        path = self.get_checkpoint_name(epoch)

        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def train(self, args, train_loader, val_loader):
        for epoch in range(self.cur_epoch, self.epochs):
            train_loss = self.train_epoch(train_loader, f'Training epoch {epoch + 1}/{self.epochs}')
            val_loss = self.train_epoch(val_loader, f'Validating epoch {epoch + 1}/{self.epochs}')
            bleu = self.calculate_bleu(val_loader, f'Calculating bleu epoch {epoch + 1}/{self.epochs}')

            # wandb.log({"train_loss": train_loss, "val_loss": val_loss, "bleu": float(bleu)})

            print(
                "epoch: {} train_loss: {:.5f} val_loss: {:.5f} bleu: {}".format(epoch + 1, train_loss, val_loss, bleu))

            if (epoch + 1) % args.save_step == 0:
                self.save_checkpoint(epoch + 1, val_loss)
