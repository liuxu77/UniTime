import os
import json
import copy
import time
import torch
import random
import numpy as np
import pandas as pd
import configparser

from models.unitime import UniTime
from data_provider.data_factory import data_provider
from engines.engine_forecasting import Engine_Forecasting

class Engine(object):
    def __init__(self, args):
        args.device = torch.device('cuda:{}'.format(args.gpu))
        model_map_dict = {
            'gpt2-small': 'gpt2',
        }
        args.model_path = model_map_dict[args.lm_pretrain_model]
        self.model = UniTime(args).to(args.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-6)
        self.args = args
        
        self._print_trainable_parameters(self.model.backbone)
        self._print_trainable_parameters(self.model)
        
        self._construct_unified_dataloaders()


    def _print_trainable_parameters(self, model):
        freeze = 0
        trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable += param.nelement()
            else:
                freeze += param.nelement()
        self.args.logger.info('Trainable Params: {}, All Params: {}, Percent: {}'.format(
                              trainable, freeze + trainable, trainable / (freeze + trainable)))


    def _construct_unified_dataloaders(self):
        f = open(self.args.instruct_path)
        instruct_list = json.load(f)
        f.close()

        if self.args.is_training:
            df = pd.read_csv(self.args.training_list)
            self.train_batches = 0
            self.train_loaders = []
            self.train_engines = []
            self.valid_loaders = []
            self.valid_engines = []
            self.test_loaders = []
            self.test_engines = []
        else:
            df = pd.read_csv(self.args.inference_list)
            self.test_loaders = []
            self.test_engines = []

        for i, row in df.iterrows():
            args = copy.deepcopy(self.args)
            data_name = row['Data']
            train_flag = row['Train']
            valid_flag = row['Valid']
            test_flag = row['Test']

            config = configparser.ConfigParser()
            config.read('data_configs/' + data_name + '.conf')
            data_config = config['config']

            args.data_path = data_config['data_path']
            args.data_reader = data_config['data_reader']
            args.data_id = data_config['data_id']
            args.features = data_config['features']
            args.seq_len = int(data_config['seq_len'])
            args.stride = int(data_config['stride'])
            args.batch_size = int(data_config['batch_size'])
            if test_flag and self.args.zero_shot_instruct != '':
                args.instruct = self.args.zero_shot_instruct
            else:
                args.instruct = instruct_list[data_name]

            args.pred_len = int(row['Prediction'])
            args.mask_rate = self.args.mask_rate
            eng = Engine_Forecasting(args)
            setting = '{}_{}_{}_{}_{}_{}_{}'.format(args.data_id, args.features, args.seq_len, args.pred_len, args.mask_rate, args.stride, args.batch_size, args.learning_rate)

            self.args.logger.info('***** Task: {} *****'.format(setting))

            if self.args.is_training:
                if train_flag:
                    _, train_loader = data_provider(args, 'train')
                    self.train_batches += len(train_loader)
                    self.train_loaders.append(train_loader)
                    self.train_engines.append(eng)
                if valid_flag:
                    _, valid_loader = data_provider(args, 'val')
                    self.valid_loaders.append(valid_loader)
                    self.valid_engines.append(eng)
                if test_flag:
                    _, test_loader = data_provider(args, 'test')
                    self.test_loaders.append(test_loader)
                    self.test_engines.append(eng)
            else:
                _, test_loader = data_provider(args, 'test')
                self.test_loaders.append(test_loader)
                self.test_engines.append(eng)


    def train(self):
        self.args.logger.info('Start training!')

        wait = 0
        best_valid_loss = np.array([5] * len(self.valid_loaders))
        for e in range(self.args.train_epochs):
            iterators = [d._get_iterator() for d in self.train_loaders]
            length = len(self.train_loaders)
            batch_cnt = [0] * length

            # train
            t1 = time.time()
            train_loss = []
            while True:
                idx = random.randint(0, length - 1)
                try:
                    loader = iterators[idx]
                    batch = next(loader)
                    loss = self.train_engines[idx].train_batch(batch, self.model, self.optimizer)
                    train_loss.append(loss)
                    batch_cnt[idx] += 1
                except StopIteration:
                    continue
                if sum(batch_cnt) >= self.train_batches:
                    break
            mtrain_loss = np.mean(train_loss)
            t2 = time.time()
            self.args.logger.info('Epoch: {}, Train Time: {:.6f}, Train Loss: {:.6f}'.format(e + 1, t2 - t1, mtrain_loss))

            # valid
            v1 = time.time()
            valid_loss = []
            for loader, eng in zip(self.valid_loaders, self.valid_engines):
                loss = eng.valid(loader, self.model)
                valid_loss.append(loss)
            valid_loss = np.array(valid_loss)
            mvalid_loss = np.mean(valid_loss)
            improve = np.sum((best_valid_loss - valid_loss) / best_valid_loss)
            v2 = time.time()
            self.args.logger.info('Epoch: {}, Valid Time: {:.6f}, Valid Loss: {:.6f}, Valid Loss Improve: {:.6f}'.format(e + 1, v2 - v1, mvalid_loss, improve))

            if improve >= 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint, 'model_s' + str(self.args.seed) + '.pth'))
                self.args.logger.info('Saving best model!')
                best_valid_loss = valid_loss
                wait = 0
            else:
                torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint, 'model_s' + str(self.args.seed) + '_e' + str(e + 1) + '.pth'))
                wait += 1
                if wait == self.args.patience:
                    self.args.logger.info('Early stop at epoch {}'.format(e + 1))
                    break

            self.scheduler.step()
            self.args.logger.info('Update learning rate to {}'.format(self.scheduler.get_last_lr()[0]))

        self.test()


    def test(self):
        self.args.logger.info('Start testing!')
        if self.args.eval_model_path != '':
            path = self.args.eval_model_path
        else:
            path = os.path.join(self.args.checkpoint, 'model_s' + str(self.args.seed) + '.pth')
        self.model.load_state_dict(torch.load(path))

        for loader, eng in zip(self.test_loaders, self.test_engines):
            eng.test(loader, self.model)
