import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unitimegpt2 import UniTimeGPT2
from transformers import GPT2Tokenizer

class FlattenHead(nn.Module):
    def __init__(self, fea_num, pred_len, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(fea_num, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class UniTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mask_rate = args.mask_rate
        self.patch_len = args.patch_len
        self.max_token_num = args.max_token_num
        self.max_backcast_len = args.max_backcast_len
        self.max_forecast_len = args.max_forecast_len
        self.logger = args.logger

        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        self.backbone = UniTimeGPT2.from_pretrained(args.model_path)
        self.backbone.transformer.h = self.backbone.transformer.h[:args.lm_layer_num]

        if args.lm_ft_type != 'full':
            if args.lm_ft_type == 'freeze':
                words = []
            elif args.lm_ft_type == 'fpt':
                words = ['ln', 'wpe']
            else:
                exit(0)
            for name, param in self.backbone.named_parameters():
                flag = 0
                for w in words:
                    if w in name: flag = 1
                if flag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                # print(name, param.shape, param.requires_grad)

        config = self.backbone.config
        self.d_model = config.n_embd

        self.feature_embedding = nn.Linear(args.patch_len, self.d_model)
        if args.mask_rate > 0:
            self.feature_projection = nn.Linear(self.d_model, self.d_model)
            self.binary_indicator_embedding = nn.Linear(args.patch_len, self.d_model)
            self.gate_w1 = nn.Linear(self.d_model, self.d_model)
            self.gate_w2 = nn.Linear(self.d_model, self.d_model)
            self.gate_sigmoid = nn.Sigmoid()

        self.ts_embed_dropout = nn.Dropout(args.ts_embed_dropout)

        self.pad_token = nn.Parameter(torch.randn(1, 1, self.d_model), requires_grad=True)

        dec_trans_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                     nhead=config.n_head,
                                                     dim_feedforward=self.d_model * 4, 
                                                     dropout=config.attn_pdrop,
                                                     layer_norm_eps=config.layer_norm_epsilon,
                                                     batch_first=True,
                                                     norm_first=True)
        self.dec_transformer = nn.TransformerEncoder(dec_trans_layer, num_layers=args.dec_trans_layer_num)

        self.dec_head = FlattenHead(fea_num=self.d_model * args.max_token_num,
                                    pred_len=args.max_backcast_len + args.max_forecast_len,
                                    head_dropout=args.dec_head_dropout)


    def generate_ts_token(self, x_inp, seq_len, stride, mask):
        if seq_len <= self.patch_len:
            ts_pad_num = self.patch_len - seq_len
        else:
            if seq_len % stride == 0:
                ts_pad_num = 0
            else:
                ts_pad_num = (seq_len // stride) * stride + self.patch_len - seq_len

        ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
        x_inp = ts_padding(x_inp)
        mask = ts_padding(mask)

        x_inp = x_inp.unfold(dimension=-1, size=self.patch_len, step=stride)
        mask = mask.unfold(dimension=-1, size=self.patch_len, step=stride)

        b, f, p, h = x_inp.shape
        x_inp = x_inp.reshape(b * f, p, h)
        x_embed = self.feature_embedding(x_inp)

        if self.mask_rate > 0:
            mask = mask.reshape(b * f, p, h)
            mask_embed = self.binary_indicator_embedding(mask)

            gate = self.gate_sigmoid(self.gate_w1(x_embed) + self.gate_w2(mask_embed))
            x_embed = gate * x_embed + (1 - gate) * mask_embed
            x_embed = self.feature_projection(x_embed)

        return self.ts_embed_dropout(x_embed), f


    def forward(self, info, x_inp, mask):
        data_id, seq_len, stride, instruct = info

        means = torch.sum(x_inp, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_inp -= means
        x_inp = x_inp.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_inp * x_inp, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_inp /= stdev

        x_inp = x_inp.transpose(1, 2)
        mask = mask.transpose(1, 2)
        x_token, n_vars = self.generate_ts_token(x_inp, seq_len, stride, mask)

        if len(instruct) > 0:
            instruct_ids = self.tokenizer(instruct, return_tensors='pt').input_ids.to(x_inp.device)
            instruct_embed = self.backbone.transformer.wte(instruct_ids).repeat(x_token.shape[0], 1, 1)
            inputs_embeds = torch.cat((instruct_embed, x_token), dim=1)
        else:
            inputs_embeds = x_token

        x_enc = self.backbone(inputs_embeds=inputs_embeds)

        bs, token_num, _ = x_enc.shape
        pad_token_num = self.max_token_num - token_num
        if pad_token_num > 0:
            p = self.pad_token.repeat(bs, pad_token_num, 1)
            x_enc = torch.cat((x_enc, p), dim=1)
        if pad_token_num < 0:
            print('Token Num Error', data_id, instruct_embed.shape, x_token.shape)
            exit(0)

        x_dec = self.dec_transformer(x_enc)
        x_dec = torch.reshape(
            x_dec, (-1, n_vars, x_dec.shape[-2], x_dec.shape[-1]))
        x_dec = x_dec.permute(0, 1, 3, 2)

        x_out = self.dec_head(x_dec)
        x_out = x_out.transpose(2, 1)

        x_out = x_out * (stdev.repeat(1, x_out.shape[1], 1))
        x_out = x_out + (means.repeat(1, x_out.shape[1], 1))
        return x_out
