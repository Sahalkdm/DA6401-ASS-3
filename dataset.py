import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
 
from datasets import load_dataset
import spacy

# Special Token indedices
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

class Vocab:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
 
    def __len__(self):
        return len(self.stoi)
 
    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)
 
    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, '<unk>')
 
    def lookup_tokens(self, indices) -> list:
        return [self.lookup_token(i) for i in indices]
 
    def get(self, token: str, default=UNK_IDX) -> int:
        return self.stoi.get(token, default)
    

def build_vocab_from_counter(counter: Counter, min_freq: int = 2) -> Vocab:
    """Build a Vocab from a word frequency counter."""
    stoi = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in counter.most_common():
        if freq >= min_freq and word not in stoi:
            stoi[word] = len(stoi)
    return Vocab(stoi)


class Multi30kDataset:
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None, min_freq=2, max_src_len=100, max_tgt_len=100, raw_data=None, nlp_de=None, nlp_en=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.split = split
        self.min_freq = min_freq
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
 
        # Reuse passed-in spaCy models, or load once
        if nlp_de is not None:
            self.nlp_de = nlp_de
        else:
            try:
                self.nlp_de = spacy.load("de_core_news_sm")
            except OSError:
                print("Downloading de_core_news_sm …")
                from spacy.cli import download as spacy_download
                spacy_download("de_core_news_sm")
                self.nlp_de = spacy.load("de_core_news_sm")
 
        if nlp_en is not None:
            self.nlp_en = nlp_en
        else:
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
            except OSError:
                print("Downloading en_core_web_sm …")
                from spacy.cli import download as spacy_download
                spacy_download("en_core_web_sm")
                self.nlp_en = spacy.load("en_core_web_sm")
 
        # Reuse passed-in HF split, or load from HuggingFace
        if raw_data is not None:
            self.raw_data = raw_data
        else:
            hf_split = "validation" if split in ("val", "validation") else split
            raw = load_dataset("bentrevett/multi30k")
            self.raw_data = raw[hf_split]
 
        # Tokenise all sentences
        self.src_tokens = [self._tokenise_de(ex['de']) for ex in self.raw_data]
        self.tgt_tokens = [self._tokenise_en(ex['en']) for ex in self.raw_data]
 
        # Build or reuse vocabulary
        if src_vocab is None:
            src_counter = Counter(tok for sent in self.src_tokens for tok in sent)
            self.src_vocab = build_vocab_from_counter(src_counter, min_freq)
        else:
            self.src_vocab = src_vocab
 
        if tgt_vocab is None:
            tgt_counter = Counter(tok for sent in self.tgt_tokens for tok in sent)
            self.tgt_vocab = build_vocab_from_counter(tgt_counter, min_freq)
        else:
            self.tgt_vocab = tgt_vocab
 
        # Convert tokens to integer sequences
        self.src_data, self.tgt_data = self._numericalize()
 
    # Tokenisers
    def _tokenise_de(self, text: str) -> list:
        return [tok.text.lower() for tok in self.nlp_de(text)]
 
    def _tokenise_en(self, text: str) -> list:
        return [tok.text.lower() for tok in self.nlp_en(text)]
 
    # Numericalization
    def _numericalize(self):
        src_data, tgt_data = [], []
        for src_toks, tgt_toks in zip(self.src_tokens, self.tgt_tokens):
            
            src_ids = ([SOS_IDX]+ [self.src_vocab[t] for t in src_toks[:self.max_src_len]]+ [EOS_IDX])
            tgt_ids = ([SOS_IDX] + [self.tgt_vocab[t] for t in tgt_toks[:self.max_tgt_len]] + [EOS_IDX])

            src_data.append(torch.tensor(src_ids, dtype=torch.long))
            tgt_data.append(torch.tensor(tgt_ids, dtype=torch.long))
        return src_data, tgt_data
  
    def __len__(self):
        return len(self.src_data)
 
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]
    
def collate_fn(batch):
    """
    Collate a list of (src, tgt) tensor pairs into padded batch tensors.
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch
 
def build_datasets(min_freq: int = 2, max_src_len: int = 100, max_tgt_len: int = 100):
    """
    Build train, val, and test Multi30k datasets with a shared vocabulary
    (built on training data only, as is correct practice).
    """
    train_ds = Multi30kDataset(
        split='train',
        min_freq=min_freq,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    val_ds = Multi30kDataset(
        split='validation',
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    test_ds = Multi30kDataset(
        split='test',
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    return train_ds, val_ds, test_ds
 
 
def build_dataloaders( train_ds, val_ds, test_ds, batch_size: int = 128, num_workers: int = 0):
    """
    Wrap datasets into DataLoader objects.
 
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
