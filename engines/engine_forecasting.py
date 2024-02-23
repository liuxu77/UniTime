import os
import torch
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from utils.metrics import metric

class Engine_Forecasting(object):
    def __init__(self, args):
        self.args = args
        self.data_id = args.data_id + '_' + str(args.seq_len) + '_' + str(args.pred_len)
        self.info = [self.data_id, args.seq_len, args.stride, args.instruct]
        self.criterion = torch.nn.MSELoss()


    def train_batch(self, batch, model, optimizer):
        model.train()
        optimizer.zero_grad()

        batch_x, batch_y = batch
        batch_x = batch_x.float().to(self.args.device)
        batch_y = batch_y.float().to(self.args.device)

        b, t, n = batch_x.shape
        mask = torch.rand((b, t, n)).to(self.args.device)
        mask[mask < self.args.mask_rate] = 0  # masked
        mask[mask >= self.args.mask_rate] = 1  # remained
        inp = batch_x.masked_fill(mask == 0, 0)

        outputs = model(self.info, inp, mask)
        f_dim = -1 if self.args.features == 'MS' else 0
        if self.args.max_backcast_len == 0:
            outputs = outputs[:, :self.args.pred_len, f_dim:]
            batch_y = batch_y[..., f_dim:]
        elif self.args.max_forecast_len == 0:
            outputs = outputs[:, self.args.max_backcast_len-self.args.seq_len:, f_dim:]
            batch_y = batch_x[..., f_dim:]
        else:
            outputs = outputs[:, self.args.max_backcast_len-self.args.seq_len:
                              self.args.max_backcast_len+self.args.pred_len, f_dim:]
            batch_y = torch.cat((batch_x, batch_y), dim=1)
            batch_y = batch_y[..., f_dim:]

        loss = self.criterion(outputs, batch_y)
        loss.backward()
        if self.args.clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        optimizer.step()
        return loss.item()


    def valid(self, valid_loader, model):
        valid_loss = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(valid_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                b, t, n = batch_x.shape
                mask = torch.ones((b, t, n)).to(self.args.device)

                outputs = model(self.info, batch_x, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, self.args.max_backcast_len:
                                  self.args.max_backcast_len+self.args.pred_len, f_dim:]
                batch_y = batch_y[..., f_dim:]

                loss = self.criterion(outputs, batch_y)
                valid_loss.append(loss.item())

        valid_loss = np.average(valid_loss)
        return valid_loss


    def test(self, test_loader, model):
        preds = []
        trues = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                b, t, n = batch_x.shape
                mask = torch.ones((b, t, n)).to(self.args.device)

                outputs = model(self.info, batch_x, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, self.args.max_backcast_len:
                                  self.args.max_backcast_len+self.args.pred_len, f_dim:]
                batch_y = batch_y[..., f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.args.logger.info('Setting: {}, MSE: {:.6f}, MAE: {:.6f}'.format(self.data_id, mse, mae))

        f = open(os.path.join(self.args.checkpoint, 'result_s' + str(self.args.seed) + '.txt'), 'a')
        f.write(self.data_id + '\n')
        f.write('MSE: {}, MAE: {}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
