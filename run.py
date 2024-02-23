import os
import random
import torch
import numpy as np
import pandas as pd
import argparse

from engines.engine import Engine
from utils.logger import get_logger
torch.set_num_threads(3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTime')

    # basic
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--training_list', type=str, default='execute_list/train_all.csv', help='list of the training tasks')
    parser.add_argument('--inference_list', type=str, default='execute_list/inference_all.csv', help='list of the inference tasks')
    parser.add_argument('--eval_model_path', type=str, default='', help='pretrain model path for evaluation')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--seed', type=int, default=2036, help='random seed')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--label_len', type=int, default=0, help='label length')

    # model
    parser.add_argument('--lm_pretrain_model', type=str, default='gpt2-small', help='pretrain model name')
    parser.add_argument('--lm_ft_type', type=str, default='full', help='fine-tuning type, options:[freeze: all parameters freeze, fpt: only tune positional embeddings and layernorms, full: full parameters tuning]')
    parser.add_argument('--instruct_path', type=str, default='data_configs/instruct.json', help='instruction list')
    parser.add_argument('--zero_shot_instruct', type=str, default='', help='zero shot instruction')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='masking rate')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--max_token_num', type=int, default=0, help='maximum token number')
    parser.add_argument('--max_backcast_len', type=int, default=96, help='maximum backcast sequence length')
    parser.add_argument('--max_forecast_len', type=int, default=720, help='maximum forecast sequence length')
    parser.add_argument('--lm_layer_num', type=int, default=6, help='language model layer number')
    parser.add_argument('--dec_trans_layer_num', type=int, default=2, help='decoder transformer layer number')
    parser.add_argument('--ts_embed_dropout', type=float, default=0.3, help='time series embedding dropout')
    parser.add_argument('--dec_head_dropout', type=float, default=0.1, help='decoder head dropout')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stop patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--clip', type=int, default=5, help='gradient clipping')

    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set logger    
    args.checkpoint = 'checkpoint_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.lm_pretrain_model.lower(), args.lm_ft_type, args.training_list.split('/')[1].split('.')[0], args.instruct_path.split('/')[1].split('.')[0], args.lm_layer_num, args.dec_trans_layer_num, args.mask_rate, args.max_backcast_len)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    logger = get_logger(args.checkpoint, __name__, 'record_s' + str(args.seed) + '.log')
    logger.info(args)
    args.logger = logger

    # set engine
    engine = Engine(args)
    if args.is_training:
        engine.train()
    else:
        engine.test()
