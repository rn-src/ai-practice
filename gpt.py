#!/bin/env python
# pylint: disable=line-too-long,too-many-instance-attributes,missing-function-docstring,missing-class-docstring,missing-module-docstring,trailing-newlines,too-many-locals,too-few-public-methods,too-many-statements,too-many-arguments,redefined-outer-name,broad-exception-caught
import sys
import math
import random
import argparse
import glob
import re
import os.path
from datetime import datetime
from dataclasses import dataclass
import torch
import torch.optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

@dataclass
class GptParams:
    "The params of a gpt"
    vocab_size: int = 50257
    emb_dims: int = 768
    context_length: int = 1024
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    hidden_states_factor: int = 4
    training_split: float = 0.9
    training_stride_ratio: float = 0.0
    training_lookahead: int = 1
    training_epochs: int = 1
    learning_rate: float = 0.0005
    weight_decay: float = 0.1
    training_batch_size: int = 1
    data_utilization: float = 1.0
    eval_interval: float = 0.2

    def suffix(self) -> str:
        return f'v{self.vocab_size}-d{self.emb_dims}-c{self.context_length}-hs{self.hidden_states_factor}-nh{self.n_heads}-nl{self.n_layers}'

class LossData:
    def __init__(self):
        self.training_epochs: list = []
        self.training_losses: list = []
        self.val_losses: list = []

class PlusEq(nn.Module):
    def __init__(self, sub: nn.Module):
        super().__init__()
        self.sub = sub

    def forward(self, x):
        return x + self.sub(x)

class Normalize(nn.Module):
    "Normalize class"
    def __init__(self, _: GptParams):
        super().__init__()

    def forward(self, x):
        "apply the function"
        return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)

class InfMask(nn.Module):
    "Infinity Mask"
    def __init__(self, x):
        super().__init__()
        self.register_buffer("mask", torch.triu(torch.ones(x,x),1).bool())

    def forward(self, x):
        return x.masked_fill_(self.mask[:x.shape[-1], :x.shape[-1]], -torch.inf)

class FeedForward(nn.Module):
    "FeedForward class"
    def __init__(self, params: GptParams):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(params.emb_dims, params.emb_dims*params.hidden_states_factor),
            nn.ReLU(),
            nn.Linear(params.emb_dims*params.hidden_states_factor, params.emb_dims),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    "MultiHeadAttention class"
    def __init__(self, params: GptParams):
        super().__init__()
        d_out = params.emb_dims
        d_in = params.emb_dims
        self.d_out = d_out
        self.params = params
        self.head_dim = d_out // params.n_heads
        self.qp = nn.Linear(d_in, d_in, bias=False)
        self.kp = nn.Linear(d_in, d_out, bias=False)
        self.vp = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=True)
        self.infmask = InfMask(params.context_length)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(params.dropout)

    def forward(self, x):
        # Supposing there are 3 batches, context_size is 1024, emb_dims = 768, 12 heads
        # then x is (3, 1024, 768)
        b, vocab_size, _ = x.shape
        # (3, 1024, 768) -> (3, 1024, 12, 64) -> (3, 12, 1024, 64) for query, keys, values
        query = self.qp(x).view(b, vocab_size, self.params.n_heads, self.head_dim).transpose(1,2)
        keys = self.kp(x).view(b, vocab_size, self.params.n_heads, self.head_dim).transpose(1,2)
        values = self.vp(x).view(b, vocab_size, self.params.n_heads, self.head_dim).transpose(1,2)
        # (3, 12, 1024, 64) x (3, 12, 64, 1024) -> (3, 12, 1024, 1024)
        attn_scores = query @ keys.transpose(2,3)
        # no change
        attn_weights = self.drop( self.softmax(self.infmask(attn_scores) / keys.shape[-1]**0.5) )
        # (3, 12, 1024, 1024) x (3, 12, 1024, 64) -> (3, 12, 1024, 64)
        context_vec = (attn_weights @ values).transpose(1,2)
        # (3, 12, 1024, 64) -> (3, 1024, 768)
        context_vec = context_vec.reshape(b, vocab_size, self.d_out)
        # (3, 1024, 768) x (768, 768) -> (3, 1024, 768)
        context_vec = self.out_proj(context_vec)
        return context_vec

class Transformer(nn.Module):
    "Transformer class"
    def __init__(self, params: GptParams):
        super().__init__()
        self.layers = nn.Sequential(
          PlusEq(
            nn.Sequential(
              Normalize(params),
              MultiHeadAttention(params),
              nn.Dropout(params.dropout),
            )
          ),
          PlusEq(
            nn.Sequential(
              Normalize(params),
              FeedForward(params),
              nn.Dropout(params.dropout),
            )
          )
        )

    def forward(self, x):
        "(+= Norm Mha Drop) (+= Norm Ff Drop)"
        return self.layers(x)

class EmbPos(nn.Module):
    def __init__(self, params: GptParams):
        super().__init__()
        i = torch.arange(0,params.emb_dims//2)
        pos = torch.arange(0,params.context_length).view(-1,1).float()
        mul = torch.exp(i * 2.0 * -math.log(10000.0) / params.emb_dims).view(1,-1)
        val = pos @ mul
        s = val.sin()
        c = val.cos()
        pes = torch.zeros(size=[params.context_length,params.emb_dims])
        for i in range(params.context_length):
            pes[i] = torch.stack((s[i],c[i])).T.reshape(1,-1)
        self.register_buffer("e_pos", pes)

    def forward(self, x):
        # x.shape is batch, tokens, emb_dim
        return self.e_pos[:x.shape[-2]]

class GptModule(nn.Module):
    "Gpt class"
    def __init__(self, params: GptParams) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Embedding(params.vocab_size, params.emb_dims),
            PlusEq(
              EmbPos(params),
            ),
            nn.Dropout(params.dropout),
            nn.Sequential(*[Transformer(params) for _ in range(params.n_layers)]),
            Normalize(params),
            nn.Linear(params.emb_dims, params.vocab_size, bias=False),
        )

    def forward(self, x):
        "(E += Epos) Drop Trns^l Norm Einv"
        return self.layers(x)

class GptDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, params: GptParams):
        self.token_ids = token_ids
        self.slice_length = params.context_length
        self.lookahead = params.training_lookahead
        self.stride = 1 if params.training_stride_ratio == 0 else int(params.training_stride_ratio*params.context_length)
        self.len = len(self.token_ids)-self.slice_length+1-self.lookahead

    def __len__(self):
        return self.len // self.stride

    def __getitem__(self, idx):
        i = idx * self.stride
        input_chunk = self.token_ids[i:i + self.slice_length]
        target_chunk = self.token_ids[i + self.lookahead: i + self.slice_length + self.lookahead]
        return input_chunk, target_chunk


pparts: list[tuple[str,str,str]] = [
              ('🗂️','🤖','📓'),
              ('⚰️','🧛','💃'),
              ('🌭','👩🏾‍🏭','🐷'),
              ('🕳️','👨🏼‍💼','📧'),
              ('🥩','👨‍🍳','🐮'),
              ('💩','🐐','🌿'),
              ('🤢','🤮','😱'),
              ('😊','💰','😫'),
              ('🍣','🔪','🐟'),
              ('🥰','😽','😔'),
              ('🪵','🪓','🌳'),
              ('😽','🐟','😿'),
             ]

def update_progress(prefix: str, epoch: int, total_epochs: int, progress: int, total: int, suffix: str) -> None:
    text_width = 25
    filled_length = int(text_width * progress // total)
    p1, p2, p3 = pparts[(epoch-1)%len(pparts)]
    bar_text = p1 * filled_length + p2 + p3 * (text_width - filled_length)
    edigits = int(math.floor(math.log10(total_epochs)))+1
    pdigits = int(math.floor(math.log10(total)))+1
    line = ('\r{prefix} epoch {epoch: >'+f'{edigits}'+'} {bar_text} {percent: >7.3f}% ({progress: >'+f'{pdigits}'+'} / {total}) {suffix}').format(
        prefix=prefix,
        epoch=epoch,
        bar_text=bar_text,
        percent=100.0*progress/total,
        progress=progress,
        total=total,
        suffix=suffix)
    print(line, end = "\r")

class Gpt:
    def __init__(self, params: GptParams):
        self.params: GptParams = params
        self.model: GptModule = GptModule(params)
        self.device_type = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
        self.device = torch.device(self.device_type)
        self.encoder = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        return torch.tensor(data=self.encoder.encode(text, allowed_special={"<|endoftext|>"}))

    def plot(self, losses: dict[str,LossData]):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        for name, loss in losses.items():
            ax1.plot(loss.training_epochs, loss.training_losses, label=f"Training loss {name}")
            ax1.plot(loss.training_epochs, loss.val_losses, linestyle="-.", label=f"Validation loss {name}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="lower right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylim(ymin=0)
        fig.tight_layout()
        plt.show()

    def train(self, training_paths: list[str]):
        if not training_paths:
            return
        losses: dict[str,LossData] = {}
        for training_path in training_paths:
            for training_file in glob.glob(training_path):
                random.seed(datetime.now().timestamp())
                random.shuffle(pparts)
                loss = self.train_one(training_file)
                losses[os.path.split(training_file)[-1]] = loss
        if losses and self.params.eval_interval:
            self.plot(losses)

    def fname(self, name: str) -> str:
        return name+'.'+self.params.suffix()+'.gpt'

    def load(self, name: str):
        try:
            state_dict = torch.load(self.fname(name), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception:
            pass

    def save(self, name: str):
        torch.save(self.model.state_dict(), self.fname(name))

    def prompt(self, starter: str, gen_length: int, show_progress=False) -> str:
        token_ids = self.encode(starter).view(1,-1)
        self.model.eval()
        if show_progress:
            print(starter, flush=True, end='')
        for _ in range(gen_length):
            logits = self.model(token_ids[:, :self.params.context_length])
            toks = torch.argmax(logits[:, -1, :], keepdim=True, dim=-1)
            token_ids = torch.concat((token_ids, toks), dim=-1)
            if show_progress:
                print(self.encoder.decode(token_ids.flatten()[-1:].tolist()), file=sys.stdout, flush=True, end='')
        if show_progress:
            print()
        return self.encoder.decode(token_ids.flatten().tolist())

    def train_one(self, fpath: str) -> LossData:
        params = self.params
        with open(fpath,'r',encoding='utf-8') as f:
            text = f.read()
            idx_util = int(math.floor(len(text)*params.data_utilization))
            text = text[:idx_util]
            parts = re.split('\n[^\n]*(?:START|END)[^P\n]*PROJECT GUTENBERG([^\n]*\n)', text)
            if len(parts) == 5:
                fname = os.path.split(fpath)[-1]
                title = parts[1].strip().removeprefix('EBOOK').strip()
                prefix = f"{(fname+' - '+title)[:40]: <40}"
            else:
                fname = os.path.split(fpath)[-1]
                prefix = f"{fname: <40}"
        suffix = ''
        token_ids = self.encode(text)
        idx_split = int(math.floor(len(token_ids)*params.training_split))
        training_dataset = GptDataset(token_ids[:idx_split],params)
        training_data = DataLoader(
            training_dataset,
            batch_size=params.training_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
            )
        validation_dataset: GptDataset
        validation_data: DataLoader
        if params.eval_interval:
            validation_dataset = GptDataset(token_ids[idx_split:],params)
            validation_data = DataLoader(
                validation_dataset,
                batch_size=params.training_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
                )
        model = self.model
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

        print("\x1b[?25l", end='')
        try:
            loss_data = LossData()
            eval_interval = int(params.eval_interval * len(training_data.dataset))
            recent_training_loss = 0.0
            recent_validation_loss = 0.0
            for epoch in range(params.training_epochs):
                if params.eval_interval:
                    model.eval()
                    loss_data.training_epochs.append(epoch)
                    recent_training_loss = self.eval_training_loss(training_data)
                    loss_data.training_losses.append(recent_training_loss)
                    recent_validation_loss = self.eval_training_loss(validation_data)
                    loss_data.val_losses.append(recent_validation_loss)
                model.train()
                batches = len(training_data.dataset)//params.training_batch_size
                total = batches*params.training_batch_size
                for i, batch in enumerate(training_data):
                    update_progress(prefix, epoch+1, params.training_epochs, (i+1) *params.training_batch_size, total, suffix)

                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        input_batch, target_batch = batch
                        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
                        logits = model(input_batch)
                        optimizer.zero_grad()
                        training_loss = F.cross_entropy(logits.flatten(0,1), target_batch.flatten())
                        training_loss.backward()
                        optimizer.step()


                    if params.eval_interval and i != 0 and (i % eval_interval == 0 or i+1 == batches):
                        model.eval()
                        loss_data.training_epochs.append(epoch+i/batches)
                        recent_training_loss = self.eval_training_loss(training_data)
                        loss_data.training_losses.append(recent_training_loss)
                        recent_validation_loss = self.eval_training_loss(validation_data)
                        loss_data.val_losses.append(recent_validation_loss)
                        model.train()
                        suffix = f'tloss = {recent_training_loss: 6.3f} vloss = {recent_validation_loss: 6.3f}'

                print()
        finally:
            print("\x1b[?25h", end='')
        return loss_data

    def eval_training_loss(self, data) -> float:
        loss_sum = 0.0
        for inb, t in data:
            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                inb, t = inb.to(self.device), t.to(self.device)
                logits = self.model(inb)
                loss_sum += float(F.cross_entropy(logits.flatten(0,1), t.flatten()))
        return loss_sum/len(data.dataset)

parser = argparse.ArgumentParser(prog='gpt', description='gpt utility')
parser.add_argument('--vocab-size', required=False, type=int, default=50257, help='embedded dimensions, default %(default)s')
parser.add_argument('--emb-dims', required=False, type=int, default=768, help='embedded dimensions, default %(default)s')
parser.add_argument('--context-length', required=False, type=int, default=1024, help='context length, default %(default)s')
parser.add_argument('--n-heads', required=False, type=int, default=12, help='number of heads of attention, default %(default)s')
parser.add_argument('--n-layers', required=False, type=int, default=12, help='number of transformer layers, default %(default)s')
parser.add_argument('--dropout', required=False, type=float, default=0.1, help='dropout for training, default %(default)s')
parser.add_argument('--hidden-states-factor', required=False, type=int, default=4, help='ratio of hidden states to inputs in the transformer layers, default %(default)s')
parser.add_argument('--split', required=False, type=float, default=0.9, help='amount of data dedicated to training (vs validation), default %(default)s')
parser.add_argument('--stride-ratio', required=False, type=float, default=0.0, help='stride/context_length for sampling data, the bigger the fewer samples, default %(default)s')
parser.add_argument('--lookahead', required=False, type=int, default=1, help='amount of data to predict, usually just 1, default %(default)s')
parser.add_argument('--epochs', required=False, type=int, default=1, help='number of epochs to train for, default %(default)s')
parser.add_argument('--learning-rate', required=False, type=float, default=0.0005, help='learning rate for AdamW optimizer, default %(default)s')
parser.add_argument('--weight-decay', required=False, type=float, default=0.1, help='weight decay for AdamW optimizer, default %(default)s')
parser.add_argument('--batch-size', required=False, type=int, default=1, help='number of samples per batch, default %(default)s')
parser.add_argument('--data-utilization', required=False, type=float, default=1.0, help='amount of data to use for training+validation, 1=100%%, default %(default)s')
parser.add_argument('--eval-interval', required=False, type=float, default=0.1, help='interval at which to record training loss for graphical presentation, default %(default)s')
parser.add_argument('--show-loss', required=False, action='store_true', help='show the loss after training, default %(default)s')
parser.add_argument('--model-name', required=False, default='default', help='name of the model to load/save, default %(default)s')
parser.add_argument('--prompt', required=False, help='text to start a prompt with')
parser.add_argument('--prompt-output-length', required=False, type=int, default=25, help='length of generated output, in tokens, if a prompt is provided, default %(default)s')
parser.add_argument('files', nargs='*', help='files to train on')

if __name__ == '__main__':
    args = parser.parse_args()
    params = GptParams(
        vocab_size=args.vocab_size,
        emb_dims=args.emb_dims,
        context_length=args.context_length,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        hidden_states_factor=args.hidden_states_factor,
        training_split=args.split if args.show_loss else 1.0,
        training_stride_ratio=args.stride_ratio,
        training_lookahead=args.lookahead,
        training_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        training_batch_size=args.batch_size,
        data_utilization=args.data_utilization,
        eval_interval=args.eval_interval if args.show_loss else 0.0,
        )
    gpt = Gpt(params)
    gpt.load(args.model_name)
    if args.files:
        gpt.train(args.files)
        gpt.save(args.model_name)
    if args.prompt:
        gpt.prompt(args.prompt, args.prompt_output_length, show_progress=True)

