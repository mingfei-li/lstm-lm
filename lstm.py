from datasets import load_dataset
from functools import partial
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
from statistics import mean
import multiprocessing
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config():
    def __init__(self):
        self.exp_name = 'wikitext-103-raw-v1'
        self.exp_id = '4'
        self.num_epochs = 100
        self.batch_size = 50
        self.num_workers = multiprocessing.cpu_count() - 1
        self.max_length = 512
        self.vocab_size = 30522 # = AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size
        self.pad_token_id = 0 # = AutoTokenizer.from_pretrained('bert-base-uncased').pad_token_id
        self.embedding_dim = 1024
        self.hidden_dim = 1024
        self.dropout_rate = 0.3
        self.num_layers = 1
        self.tie_weights = True
        self.lr = 1e-3
        self.lr_end_factor = 1e-3
        self.grad_clip = None

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f'using {self.device}')

        self.base_dir = f'results/{self.exp_name}/{self.exp_id}'
        self.log_dir = f'{self.base_dir}/logs'
        self.model_dir = f'{self.base_dir}/models'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.model_dir)

class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
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
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)
        if config.tie_weights:
            self.fc.weight = self.embedding.weight
    
    def forward(self, x, hc=None):
        x = self.dropout(self.embedding(x))
        if hc is not None:
            x, hc = self.lstm(x, hc)
        else:
            x, hc = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hc

class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.input_gate = nn.Linear(
            in_features=2*config.hidden_dim,
            out_features=config.hidden_dim,
        )
        self.forget_gate = nn.Linear(
            in_features=2*config.hidden_dim,
            out_features=config.hidden_dim,
        )
        self.output_gate = nn.Linear(
            in_features=2*config.hidden_dim,
            out_features=config.hidden_dim,
        )
        self.cell_gen = nn.Linear(
            in_features=2*config.hidden_dim,
            out_features=config.hidden_dim,
        )
        self.fc = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.device = config.device
        if config.tie_weights:
            self.fc.weight = self.embedding.weight
    
    def forward(self, x, hc=None):
        x = x.transpose(0, 1)
        x = self.dropout(self.embedding(x))
        sequence_length, batch_size, hidden_dim = x.shape
        if hc is None:
            h = torch.zeros(batch_size, hidden_dim, device=self.device)
            c = torch.zeros(batch_size, hidden_dim, device=self.device)
        else:
            h, c = hc

        y = []
        for t in range(sequence_length):
            hx = torch.cat((h, x[t]), dim=1)
            i = F.sigmoid(self.input_gate(hx))
            f = F.sigmoid(self.forget_gate(hx))
            o = F.sigmoid(self.output_gate(hx))
            cc = F.tanh(self.cell_gen(hx))
            c = f * c + i * cc
            h = o * F.tanh(c)
            hc = (h, c)
            y.append(h)
        
        x = torch.stack(y)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x, hc

def collate(max_length, batch):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch = [doc['text'] for doc in batch]
    encoded_inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return encoded_inputs['input_ids']

def train():
    config = Config()

    ds_train = load_dataset('Salesforce/wikitext', config.exp_name)['train']
    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=partial(collate, config.max_length)
    )

    model = LSTM(config).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.lr,
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=config.lr_end_factor,
        total_iters=config.num_epochs * len(dl_train)
    )
    logger = SummaryWriter(log_dir=config.log_dir)
    step = 0
    for epoch in tqdm(range(config.num_epochs), 'Epoch: '):
        model.train()
        for batch, token_ids in enumerate(tqdm(dl_train, 'Batch: ')):
            token_ids = token_ids.to(config.device)
            input_ids = token_ids[:,:-1]
            targets = token_ids[:,1:]

            logits, _ = model(input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            lr_scheduler.step()

            grad_norm = 0.0
            for param in model.parameters():
                param_norm = param.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5

            step += config.batch_size
            logger.add_scalar('training_loss', loss.item(), step)
            logger.add_scalar('grad_norm', grad_norm, step)
            logger.add_scalar('lr', lr_scheduler.get_last_lr()[0], step)
            logger.flush()

        model.eval()
        validate(config, model, criterion, logger, step)
        generate(config, model, 'Think about', 1.0, logger, step)
        torch.save(
            model.state_dict(),
            f'{config.model_dir}/checkpoint-{epoch}.pt',
        )

def validate(config, model, criterion, logger, step):
    ds_val = load_dataset('Salesforce/wikitext', config.exp_name)['validation']
    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=partial(collate, config.max_length)
    )

    perplexity = []
    for batch, token_ids in enumerate(tqdm(dl_val, 'Val Batch: ')):
        token_ids = token_ids.to(config.device)
        input_ids = token_ids[:,:-1]
        targets = token_ids[:,1:]

        with torch.no_grad():
            logits, _ = model(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = criterion(logits, targets)
        perplexity.append(torch.exp(loss).item())
        
    perplexity = mean(perplexity)
    logger.add_scalar('perplexity', perplexity, step)
    logger.flush()

def generate(config, model, prompt, temperature, logger=None, step=None):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    token_ids = tokenizer.encode(prompt)
    assert token_ids[-1] == tokenizer.sep_token_id
    token_ids = token_ids[:-1]
    input_ids = torch.tensor(token_ids, device=config.device).unsqueeze(dim=0)

    model.eval()
    with torch.no_grad():
        output_ids, hc_states = model(input_ids)
    while len(token_ids) < config.max_length:
        logits = output_ids[0, -1] / temperature
        next_token = Categorical(logits=logits).sample()
        token_ids.append(next_token)
        if next_token == tokenizer.sep_token_id:
            break
        with torch.no_grad():
            output_ids, hc_states = model(
                torch.tensor([[next_token]], device=config.device),
                hc_states,
            )
    result = tokenizer.decode(token_ids)
    if logger is not None:
        logger.add_text(f'generated_text_temperature={temperature}', result, step)
    else:
        print(tokenizer.decode(token_ids))

def inference():
    config = Config()
    model = LSTM(config)
    model.load_state_dict(torch.load('model.pt', map_location=config.device))
    model.to(config.device)
    generate(config, model, 'San Francisco is', 0.8)
        
if __name__ == '__main__':
    train()
    # inference()
