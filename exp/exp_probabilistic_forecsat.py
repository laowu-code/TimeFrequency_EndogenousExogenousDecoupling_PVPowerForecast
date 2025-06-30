import csv
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import metrics_of_pv
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from data.data_loader import data_detime, split_data_cnn
import torch
import torch.nn as nn
from utils.loss import ql_loss,gauss_loss,lpls_loss,student_t_loss,DER,SDER
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas

warnings.filterwarnings('ignore')


class Exp_Probabilistic_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Probabilistic_Forecast, self).__init__(args)
        self.criterion = self._select_criterion(self.args.loss).to(self.device)
        self.train_loader, self.valid_loader, self.test_loader, self.scalar = self._get_data()

    def _build_model(self):
        model = self.model_dict[self.args.model_name].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        file_path=f'./data/{self.args.site}/{self.args.dataset}_{self.args.site}_2019_2022_h.csv'
        df_data = pd.read_csv(file_path, header=0)
        data_train, data_valid, data_test, timestamp_train, timestamp_valid, timestamp_test, scalar = split_data_cnn(
            df_data, self.args.rate_train, self.args.rate_test, self.args.seq_len)
        dataset_train = data_detime(data=data_train, lookback_length=self.args.seq_len,
                                    multi_steps=self.args.multi_steps,
                                    lookforward_length=self.args.pred_len)
        dataset_valid = data_detime(data=data_valid, lookback_length=self.args.seq_len,
                                    multi_steps=self.args.multi_steps,
                                    lookforward_length=self.args.pred_len)
        dataset_test = data_detime(data=data_test, lookback_length=self.args.seq_len,
                                   multi_steps=self.args.multi_steps,
                                   lookforward_length=self.args.pred_len)

        train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_valid, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=self.args.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader, scalar

    # def _select_optimizer(self):
    #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    #     return model_optim

    def _select_criterion(self, loss_name='ql'):
        if loss_name == 'ql':
            return ql_loss
        elif loss_name == 'lpls':
            return lpls_loss
        if loss_name == 'gauss':
            return gauss_loss
        elif loss_name == 'student_t':
            return student_t_loss
        if loss_name == 'DER':
            return DER
        elif loss_name == 'SDER':
            return SDER
        else:
            print('Unknown loss')
            return None

    def train(self):
        self.model.train()
        running_loss = 0.0
        for x, y in tqdm(self.train_loader):
            self.model.zero_grad()
            x, y = x.float().to(self.device), y.float().to(self.device)
            self.optim.zero_grad()
            y_pre = self.model(x)
            loss = self.criterion(y_pre, y)
            loss.backward()
            self.optim.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        #
        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        #
        # time_now = time.time()
        #
        # train_steps = len(train_loader)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        #
        # model_optim = self._select_optimizer()
        # criterion = self._select_criterion(self.args.loss)
        # mse = nn.MSELoss()
        #
        # for epoch in range(self.args.train_epochs):
        #     iter_count = 0
        #     train_loss = []
        #
        #     self.model.train()
        #     epoch_time = time.time()
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        #         iter_count += 1
        #         model_optim.zero_grad()
        #         batch_x = batch_x.float().to(self.device)
        #
        #         batch_y = batch_y.float().to(self.device)
        #         batch_y_mark = batch_y_mark.float().to(self.device)
        #
        #         # decoder input
        #         dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        #         dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        #
        #         outputs = self.model(batch_x, None, dec_inp, None)
        #
        #         f_dim = -1 if self.args.features == 'MS' else 0
        #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
        #         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        #
        #         batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
        #         loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
        #         loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
        #         loss = loss_value  # + loss_sharpness * 1e-5
        #         train_loss.append(loss.item())
        #
        #         if (i + 1) % 100 == 0:
        #             print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        #             speed = (time.time() - time_now) / iter_count
        #             left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
        #             print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
        #             iter_count = 0
        #             time_now = time.time()
        #
        #         loss.backward()
        #         model_optim.step()
        #
        #     print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        #     train_loss = np.average(train_loss)
        #     vali_loss = self.vali(train_loader, vali_loader, criterion)
        #     test_loss = vali_loss
        #     print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        #         epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        #     early_stopping(vali_loss, self.model, path)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        #
        #     adjust_learning_rate(model_optim, epoch + 1, self.args)
        #
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        #
        # return self.model

    def evaluate(self, flag='valid'):
        self.model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        data = self.test_loader if flag == 'test' else self.valid_loader
        for x, y in tqdm(data):
            self.model.zero_grad()
            with torch.no_grad():
                x, y = x.float().to(self.device), y.float().to(self.device)
                y_pre = self.model(x)
                loss = self.criterion(y_pre, y)
                val_running_loss += loss.item() * x.size(0)
                all_preds.extend(y_pre.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        epoch_loss = val_running_loss / len(data.dataset)
        if self.scalar is not None:
            all_preds = self.scalar.inverse_transform(all_preds)
            all_labels = self.scalar.inverse_transform(all_labels)
        metrics_ = metrics_of_pv(all_preds, all_labels)
        return epoch_loss, metrics_

    def run(self):
        model_name = f"{self.args.model_name}_{self.args.dataset}"
        need_train = self.args.is_training
        # need_train = False
        model_save_path = f"model_save/{self.args.site}/{self.args.dataset}" if need_train else f"model_save/{self.args.site}/{self.args.dataset}/best"
        model_save=model_save_path+'/'+f'{model_name}.pt'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if need_train:
            try:
                train_losses, valid_losses = [], []
                earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)
                for epoch in range(self.args.epochs):
                    time_start = time.time()
                    train_loss = self.train()
                    valid_loss, ms = self.evaluate()
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    self.optim_schedule.step(valid_loss)
                    earlystopping(valid_loss, self.model)
                    # torch.save(model, model_save, pickle_module=dill)
                    print('')
                    print(
                        f'epoch:{epoch+1}|{model_name}|time:{(time.time() - time_start):.2f}|Loss_train:{train_loss:.4f}|Learning_rate:{self.optim.state_dict()["param_groups"][0]["lr"]:.4f}\n'
                        f'Loss_valid:{valid_loss:.4f}|MAE:{ms[0]:.4f}|RMSE:{ms[1]:.4f}|R2:{ms[2]:.4f}|MBE:{ms[3]:.4f}',
                        flush=True)
                    if earlystopping.early_stop:
                        print("Early stopping")
                        break  # 跳出迭代，结束训练
            except KeyboardInterrupt:
                print("Training interrupted by user")
            plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
            plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid rmse")
            plt.legend()  # 显示图例
            plt.xlabel("epoches")
            # plt.ylabel("epoch")
            plt.title("Train_loss&Valid_loss")
            plt.show()

        with open(model_save, "rb") as f:
            self.model.load_state_dict(torch.load(f))
        print(self.model)
        self.model = self.model.to(self.device)
        test_loss, ms_test = self.evaluate(flag='test')
        print(
            f'Test_valid:{test_loss:.4f}|MAE:{ms_test[0]:.4f}|RMSE:{ms_test[1]:.4f}|R2:{ms_test[2]:.4f}|MBE:{ms_test[3]:.4f}', )
        if not os.path.exists(f'data_record/{self.args.site}/{self.args.dataset}'):
            os.makedirs(f'data_record/{self.args.site}/{self.args.dataset}')
        with open(f'data_record/{self.args.site}/{self.args.dataset}/Metrics_{model_name}.csv', 'a', encoding='utf-8', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([f'{self.args.site}_pred1_{model_name}', ms_test[0], ms_test[1], ms_test[2], ms_test[3]])

    # def vali(self, train_loader, vali_loader, criterion):
    #     x, _ = train_loader.dataset.last_insample_window()
    #     y = vali_loader.dataset.timeseries
    #     x = torch.tensor(x, dtype=torch.float32).to(self.device)
    #     x = x.unsqueeze(-1)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         # decoder input
    #         B, _, C = x.shape
    #         dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
    #         dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
    #         # encoder - decoder
    #         outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
    #         id_list = np.arange(0, B, 500)  # validation set size
    #         id_list = np.append(id_list, B)
    #         for i in range(len(id_list) - 1):
    #             outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
    #                                                                   dec_inp[id_list[i]:id_list[i + 1]],
    #                                                                   None).detach().cpu()
    #         f_dim = -1 if self.args.features == 'MS' else 0
    #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #         pred = outputs
    #         true = torch.from_numpy(np.array(y))
    #         batch_y_mark = torch.ones(true.shape)
    #
    #         loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)
    #
    #     self.model.train()
    #     return loss

    # def test(self, setting, test=0):
    #     _, train_loader = self._get_data(flag='train')
    #     _, test_loader = self._get_data(flag='test')
    #     x, _ = train_loader.dataset.last_insample_window()
    #     y = test_loader.dataset.timeseries
    #     x = torch.tensor(x, dtype=torch.float32).to(self.device)
    #     x = x.unsqueeze(-1)
    #
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         B, _, C = x.shape
    #         dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
    #         dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
    #         # encoder - decoder
    #         outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
    #         id_list = np.arange(0, B, 1)
    #         id_list = np.append(id_list, B)
    #         for i in range(len(id_list) - 1):
    #             outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
    #                                                                   dec_inp[id_list[i]:id_list[i + 1]], None)
    #
    #             if id_list[i] % 1000 == 0:
    #                 print(id_list[i])
    #
    #         f_dim = -1 if self.args.features == 'MS' else 0
    #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #         outputs = outputs.detach().cpu().numpy()
    #
    #         preds = outputs
    #         trues = y
    #         x = x.detach().cpu().numpy()
    #
    #         for i in range(0, preds.shape[0], preds.shape[0] // 10):
    #             gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
    #             pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
    #             visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    #
    #     print('test shape:', preds.shape)
    #
    #     # result save
    #     folder_path = './m4_results/' + self.args.model + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
    #     forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
    #     forecasts_df.index.name = 'id'
    #     forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
    #     forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')
    #
    #     print(self.args.model)
    #     file_path = './m4_results/' + self.args.model + '/'
    #     if 'Weekly_forecast.csv' in os.listdir(file_path) \
    #             and 'Monthly_forecast.csv' in os.listdir(file_path) \
    #             and 'Yearly_forecast.csv' in os.listdir(file_path) \
    #             and 'Daily_forecast.csv' in os.listdir(file_path) \
    #             and 'Hourly_forecast.csv' in os.listdir(file_path) \
    #             and 'Quarterly_forecast.csv' in os.listdir(file_path):
    #         m4_summary = M4Summary(file_path, self.args.root_path)
    #         # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
    #         smape_results, owa_results, mape, mase = m4_summary.evaluate()
    #         print('smape:', smape_results)
    #         print('mape:', mape)
    #         print('mase:', mase)
    #         print('owa:', owa_results)
    #     else:
    #         print('After all 6 tasks are finished, you can calculate the averaged index')
    #     return
