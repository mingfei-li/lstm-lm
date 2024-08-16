from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

class Config():
    def __init__(self):
        self.batch_size = 100
        self.num_workers = 0
        self.max_length = 512
        self.vocab_size = AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size
        self.embedding_dim = 100
        self.hidden_dim = 100
        self.dropout_rate = 0.65
        self.num_layers = 1
        self.tie_weights = True

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')


class RNN(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            dropout=config.dropout_rate,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=vocab_size,
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)
        if config.tie_weights:
            assert self.embedding_dim == self.hidden_dim
            self.fc.weight = self.embedding.weight
    
    def forward(x):
        x = self.dropout(self.embedding(x))
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def collate(max_length, batch):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(
        batch['text'],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask']

def train():
    config = Config()

    ds_train = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate, max_length)
    )

    model = RNN(config)
    for token_ids, mask in tqdm(dl_train):
        model.train()
        targets = token_ids[:,1:]
        mask = mask_[:,1:]
        input_ids = token_ids[:,:-1]
        logits = model(token_ids)
        torch.gather
    

if __name__ == '__main__':
    train()