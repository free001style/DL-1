import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence


class Dataset(Dataset):
    UNK_IDX = 0
    PAD_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, args, mode, rebuild=False):
        super().__init__()

        assert not (rebuild and mode != "train"), "You can rebuild vocab only within train dataset"

        self.mode = mode

        paths = []
        source_path = self.dataset_path_from_config(args, mode, "source")
        paths.append(source_path)

        if not self.is_test:
            target_path = self.dataset_path_from_config(args, mode, "target")
            paths.append(target_path)

        self.source_language = args.source_language
        self.target_language = None

        if not self.is_test:
            self.target_language = args.target_language

        self.texts = {}

        for ln, path in zip(self.get_languages(), paths):
            self.texts[ln] = self.texts_from_dataset(path)

        if not self.is_test:
            assert len(self.texts[self.source_language]) == len(
                self.texts[self.target_language]), "Size of src and dst datasets must match"

        self.tokenizers = {}

        for ln in self.get_languages():
            self.tokenizers[ln] = self.create_tokenizer(ln)

        vocab_path = self.vocab_path_from_config(args)

        self.vocabs = {}

        if os.path.exists(vocab_path) and not rebuild:
            self.vocabs = torch.load(vocab_path)
        else:
            for ln in self.get_languages():
                self.vocabs[ln] = Dataset.create_vocab(self.tokenizers[ln], self.texts[ln], args.min_freq)

            if not os.path.exists(os.path.dirname(vocab_path)):
                os.mkdir(os.path.dirname(vocab_path))
            torch.save(self.vocabs, vocab_path)

        self.transforms = {}
        for ln in self.get_languages():
            self.transforms[ln] = self.sequential_transforms(
                self.tokenizers[ln], self.vocabs[ln], Dataset.tensor_transform
            )

    @property
    def is_test(self):
        return self.mode == "test"

    @staticmethod
    def texts_from_dataset(dataset_path):
        with open(dataset_path, encoding="utf-8") as f:
            return [line.rstrip() for line in f.readlines()]

    @staticmethod
    def create_tokenizer(language):
        return get_tokenizer(None, language=language)

    @staticmethod
    def yield_token(tokenizer, texts):
        for line in tqdm(texts, desc="Building vocab"):
            sentence = tokenizer(line)
            for token in sentence:
                yield token

    @staticmethod
    def create_vocab(tokenizer, texts, min_freq):
        special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

        vocab = build_vocab_from_iterator([Dataset.yield_token(tokenizer, texts)],
                                          min_freq=min_freq, special_first=True, specials=special_symbols)
        vocab.set_default_index(Dataset.UNK_IDX)

        return vocab

    @property
    def src_vocab(self):
        return self.vocabs[self.source_language]

    @property
    def dst_vocab(self):
        assert not self.is_test, "There is no dst vocab for test dataset"
        return self.vocabs[self.target_language]

    def get_languages(self):
        if not self.is_test:
            return self.source_language, self.target_language
        return (self.source_language,)

    def __len__(self):
        return len(self.texts[self.source_language])

    @staticmethod
    def tensor_transform(indices):
        return torch.cat((torch.tensor([Dataset.BOS_IDX]),
                          torch.tensor(indices),
                          torch.tensor([Dataset.EOS_IDX])))

    def getitem(self, index, language):
        return self.transforms[language](self.texts[language][index])

    def __getitem__(self, index):
        return tuple(self.getitem(index, ln) for ln in self.get_languages())

    @staticmethod
    def get_collate_fn(is_test=False):
        def collate_fn(batch):
            source_lengths = []
            for b in batch:
                source_lengths.append(b[0].shape[0])
            src_batch = pad_sequence([b[0] for b in batch], padding_value=Dataset.PAD_IDX, batch_first=True)

            if is_test:
                return src_batch, None

            target_lengths = []
            for b in batch:
                target_lengths.append(b[1].shape[0])
            dst_batch = pad_sequence([b[1] for b in batch], padding_value=Dataset.PAD_IDX, batch_first=True)

            return src_batch, torch.tensor(source_lengths), dst_batch, torch.tensor(target_lengths)

        return collate_fn

    @staticmethod
    def vocab_path_from_config(args):
        return os.path.join(args.vocab_dir, "vocab.pth")

    @staticmethod
    def dataset_path_from_config(args, mode, type):
        if mode == 'train':
            if type == 'source':
                return os.path.join(args.datadir, args.train_source)
            elif type == 'target':
                return os.path.join(args.datadir, args.train_target)
            else:
                raise ValueError()
        elif mode == 'val':
            if type == 'source':
                return os.path.join(args.datadir, args.val_source)
            elif type == 'target':
                return os.path.join(args.datadir, args.val_target)
            else:
                raise ValueError()
        elif mode == 'test':
            return os.path.join(args.datadir, args.test_source)

    @staticmethod
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func
