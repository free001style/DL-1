from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from translate import translate
from train import Trainer
import warnings
import argparse
import numpy as np
import random
import wandb

# wandb.login()

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    set_random_seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    train_dataset = Dataset(args, "train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=Dataset.get_collate_fn(),
                              pin_memory=True if torch.cuda.is_available() else False, generator=g)

    val_dataset = Dataset(args, "val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=Dataset.get_collate_fn(),
                            pin_memory=True if torch.cuda.is_available() else False, generator=g)

    vocab_sizes = len(train_dataset.src_vocab), len(train_dataset.dst_vocab)
    target_vocab = train_dataset.vocabs["en"]
    trainer = Trainer(args, vocab_sizes, target_vocab, args.exp_name)
    # wandb.init(
    #     project='BHW-2',
    #     entity='free001style',
    #     name=args.exp_name,
    #     config=args
    # )
    trainer.train(args, train_loader, val_loader)
    # wandb.finish()

    trainer.load_from_checkpoint(epoch=args.epochs)

    test_dataset = Dataset(args, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=Dataset.get_collate_fn(is_test=True), generator=g)
    target_vocab = train_dataset.vocabs["en"]

    translated = []

    for source, _ in tqdm(test_loader):
        predict_tokens = list(translate(trainer.model, source[0], device).cpu().numpy())
        if args.unk_replace:
            sentence = " ".join(target_vocab.lookup_tokens(predict_tokens)).replace("<bos>", "").replace("<eos>",
                                                                                                         "").replace(
                "<unk>",
                "").strip()
        else:
            sentence = " ".join(target_vocab.lookup_tokens(predict_tokens)).replace("<bos>", "").replace("<eos>",
                                                                                                         "").strip()
        translated.append(sentence)

    with open(args.output, "w") as f:
        for sentence in translated:
            f.write(sentence + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="ёбанный дл")
    ############################# language ################################
    parser.add_argument("--source_language", type=str, default="de")
    parser.add_argument("--target_language", type=str, default="en")
    ############################# data ####################################
    parser.add_argument("--datadir", type=str, default="data")
    parser.add_argument("--vocab_dir", type=str, default="vocab")
    parser.add_argument("--train_source", type=str, default="train.de-en.de")
    parser.add_argument("--train_target", type=str, default="train.de-en.en")
    parser.add_argument("--val_source", type=str, default="val.de-en.de")
    parser.add_argument("--val_target", type=str, default="val.de-en.en")
    parser.add_argument("--test_source", type=str, default="test1.de-en.de")
    parser.add_argument("--min_freq", type=int, default=5)

    ############################# model ####################################
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--feedforward_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    ######################### optimizer ####################################
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.98))
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--label_smoothing", type=float, default=0.)
    ############################# checkpoint ##############################
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--save_step", type=int, default=1)

    parser.add_argument("--output", type=str, default='predict.txt')
    parser.add_argument("--unk_replace", type=bool, default=True)

    args = parser.parse_args()
    main(args)
