import argparse
import csv
import time
import pandas as pd
import dill
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
# from models.iTransformer import iTransformer_single, iTransformer_block
# from models import KAN
# import models
from data import split_data, data_detime
from utils.tools import metrics_of_pv, EarlyStopping, same_seeds, train, evaluate
from utils import ql_loss
from models import TCN, Dlinear, FreTS, LSTM, GRU, Pyraformer, iTransformer_single, PatchTST, TimeXer, Proposed
import os
import warnings
import json
from utils.tools import save_dict_to_excel
import optuna
import warnings
from utils import visualize_feature_map
from matplotlib import rc
# from matplotlib.font_manager import FontProperties
# # from matplotlib.ticker import StrMethodFormatter
rc('font', family='Arial')
# plt.rcParams['font.family'] = 'SimHei'  # 指定中文字体为黑体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 英文使用 Arial
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')


# ['TCN','Dlinear']
def select_hyperparameters(trial, model_name='Proposed', type='single', seq_len=48, pred_len=4, enc_in=5,
                           dataset='2019_2023_1h', site='site_1B'):
    params_base = {'seq_len': seq_len, 'pred_len': pred_len, 'enc_in': enc_in, }
    if type == 'optimal':
        if model_name == 'TCN':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'channels': 16, 'e_layers': 3, 'kernel_size': 2}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'channels': 16, 'e_layers': 2, 'kernel_size': 4}
            elif dataset == '01_15min':
                params = {'channels': 16, 'e_layers': 3, 'kernel_size': 3}
            elif dataset == '03_15min':
                params = {'channels': 16, 'e_layers': 2, 'kernel_size': 3}
            else:
                raise ValueError('Dataset not found')

        elif model_name == 'Dlinear':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'moving_avg': 4, "individual": False}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'moving_avg': 4, "individual": False}
            elif dataset == '01_15min':
                params = {'moving_avg': 4, "individual": False}
            elif dataset == '03_15min':
                params = {'moving_avg': 4, "individual": False}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'FreTS':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'embed_size': 32, 'hidden_size': 256}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'embed_size': 16, 'hidden_size': 64}
            elif dataset == '01_15min':
                params = {'embed_size': 16, 'hidden_size': 256}
            elif dataset == '03_15min':
                params = {'embed_size': 16, 'hidden_size': 256}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'LSTM':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 64, 'e_layers': 1}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 32, 'e_layers': 2}
            elif dataset == '01_15min':
                params = {'d_model': 16, 'e_layers': 1}
            elif dataset == '03_15min':
                params = {'d_model': 64, 'e_layers': 3}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'GRU':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 16, 'e_layers': 2}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 32, 'e_layers': 1}
            elif dataset == '01_15min':
                params = {'d_model': 32, 'e_layers': 2}
            elif dataset == '03_15min':
                params = {'d_model': 63, 'e_layers': 3}
            else:
                raise ValueError('Dataset not found')

        elif model_name == 'Pyraformer':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'n_heads': 6, 'e_layers': 2, 'd_model': 32}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'n_heads': 6, 'e_layers': 1, 'd_model': 256}
            elif dataset == '01_15min':
                params = {'n_heads': 8, 'e_layers': 2, 'd_model': 64}
            elif dataset == '03_15min':
                params = {'n_heads': 8, 'e_layers': 1, 'd_model': 128}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'PatchTST':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 256, 'n_heads': 4, 'e_layers': 2, 'patch_len': 4, 'stride_flag': 'half'}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 128, 'n_heads': 6, 'e_layers': 1, 'patch_len': 8, 'stride_flag': 'full'}
            elif dataset == '01_15min':
                params = {'d_model': 128, 'n_heads': 6, 'e_layers': 1, 'patch_len': 4, 'stride_flag': 'half'}
            elif dataset == '03_15min':
                params = {'d_model': 128, 'n_heads': 8, 'e_layers': 2, 'patch_len': 2, 'stride_flag': 'half'}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'iTransformer':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 16, 'e_layers': 1, 'n_heads': 4}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 64, 'e_layers': 1, 'n_heads': 6}
            elif dataset == '01_15min':
                params = {'d_model': 16, 'e_layers': 1, 'n_heads': 4}
            elif dataset == '03_15min':
                params = {'d_model': 16, 'e_layers': 4, 'n_heads': 4}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'TimeXer':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 256, 'enc_in': 1, 'n_heads': 8, 'e_layers': 1, 'patch_len': 8}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 64, 'enc_in': 1, 'n_heads': 4, 'e_layers': 3, 'patch_len': 8}
            elif dataset == '01_15min':
                params = {'d_model': 256, 'enc_in': 1, 'n_heads': 6, 'e_layers': 1, 'patch_len': 6}
            elif dataset == '03_15min':
                params = {'d_model': 32, 'enc_in': 1, 'n_heads': 8, 'e_layers': 3, 'patch_len': 2}
            else:
                raise ValueError('Dataset not found')
        elif model_name == 'Proposed':
            if site == 'site_1B' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 512, 'n_heads': 6, 'e_layers': 3, "embed_size": 16, 'hidden_size': 32,
                          'patch_len': 8}
            elif site == 'site_24' and dataset == '2021_3_2021_5_5min':
                params = {'d_model': 128, 'n_heads': 6, 'e_layers': 3, "embed_size": 32, 'hidden_size': 32,
                          'patch_len': 4}
            elif dataset == '01_15min':
                params = {'d_model': 64, 'n_heads': 6, 'e_layers': 1, "embed_size": 16, 'hidden_size': 16,
                          'patch_len': 4}
            elif dataset == '03_15min':
                params = {'d_model': 64, 'n_heads': 6, 'e_layers': 1, "embed_size": 16, 'hidden_size': 32,
                          'patch_len': 4}
            else:
                raise ValueError('Dataset not found')
        else:
            raise ValueError('Model name not found')

    elif type == 'single':
        if model_name == 'Proposed':
            params = {'d_model': 256, 'n_heads': 6, 'e_layers': 3, "embed_size": 16, 'hidden_size': 32, 'patch_len': 2}
        elif model_name == 'Dlinear':
            params = {'moving_avg': 4, "individual": False}
        elif model_name == 'FreTS':
            params = {'embed_size': 128, 'hidden_size': 256}
        elif model_name == 'LSTM':
            params = {'d_model': 128, 'e_layers': 1}
        elif model_name == 'GRU':
            params = {'d_model': 128, 'e_layers': 1}
        elif model_name == 'TCN':
            params = {'channels': 128, 'e_layers': 3, 'kernel_size': 3}
        elif model_name == 'Pyraformer':
            params = {'n_heads': 8, 'e_layers': 2, 'd_model': 64}
        elif model_name == 'iTransformer':
            params = {'d_model': 16, 'e_layers': 3, 'n_heads': 8}
        elif model_name == 'PatchTST':
            params = {'d_model': 128, 'n_heads': 4, 'e_layers': 1, 'patch_len': 6, 'stride_flag': 'half'}
        elif model_name == 'TimeXer':
            params = {'d_model': 256, 'enc_in': 1, 'n_heads': 4, 'e_layers': 3, 'patch_len': 6}
        else:
            raise ValueError('Model name not found')
    elif type == 'optuna':
        if model_name == 'Proposed':
            params = {'d_model': trial.suggest_categorical('d_model', [32, 64, 128, 256]),
                      'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8, 12]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, ]),
                      "embed_size": trial.suggest_categorical('embed_size', [16, 32, 64, 128]),
                      'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128, ]),
                      'patch_len': trial.suggest_categorical('patch_len', [2, 4, 6, 8]), }
        elif model_name == 'Dlinear':
            params = {'moving_avg': trial.suggest_categorical('moving_avg', [2, 4, 6, ]),
                      "individual": trial.suggest_categorical('individual', [True, False])}
        elif model_name == 'FreTS':
            params = {'embed_size': trial.suggest_categorical('embed_size', [16, 32, 64, 128, 256]),
                      'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])}
        elif model_name == 'LSTM':
            params = {'d_model': trial.suggest_categorical('d_model', [16, 32, 64, 128, 256]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4])}
        elif model_name == 'GRU':
            params = {'d_model': trial.suggest_categorical('d_model', [16, 32, 64, 128, 256]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, ])}
        elif model_name == 'TCN':
            params = {'channels': trial.suggest_categorical('channels', [16, 32, 64, 128]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, ]),
                      'kernel_size': trial.suggest_categorical('kernel_size', [2, 3, 4, ])}
        elif model_name == 'Pyraformer':
            params = {
                'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8]),
                'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4]),
                'd_model': trial.suggest_categorical('d_model', [16, 32, 64, 128])}
        elif model_name == 'iTransformer':
            params = {'d_model': trial.suggest_categorical('d_model', [16, 32, 64, 128, ]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, ]),
                      'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8]), }
        elif model_name == 'PatchTST':
            params = {'d_model': trial.suggest_categorical('d_model', [16, 32, 64, 128]),
                      'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8, ]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, ]),
                      'patch_len': trial.suggest_categorical('patch_len', [2, 4, 6, 8]),
                      'stride_flag': trial.suggest_categorical('stride_flag', ['full', 'half'])}
        elif model_name == 'TimeXer':
            params = {'d_model': trial.suggest_categorical('d_model', [16, 32, 64, 128]),
                      'n_heads': trial.suggest_categorical('n_heads', [4, 6, 8]),
                      'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, ]),
                      'patch_len': trial.suggest_categorical('patch_len', [2, 4, 6, 8])}
        else:
            raise ValueError('Model name not found')
    else:
        raise ValueError('Type not found')

    params = {**params_base, **params}
    return params


def save_metrics_to_excel(metrics_dict, file_path, sheet_name='Metrics'):
    """
    Save multi-step prediction metrics to an Excel file.

    Parameters:
    metrics_dict (dict): Dictionary containing metrics, with lists for each step.
    file_path (str): Path to save the Excel file.
    sheet_name (str): Name of the sheet where the data should be saved.

    Returns:
    None
    """
    df = pd.DataFrame(metrics_dict)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a' if os.path.exists(file_path) else 'w') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    # print(f"Metrics saved to {file_path} (Sheet: {sheet_name})")


def worker_init_fn(worker_id):
    np.random.seed(seeds + worker_id)


if __name__ == "__main__":
    model_list = {'TCN': TCN, 'FreTS': FreTS, 'LSTM': LSTM, 'GRU': GRU, 'Pyraformer': Pyraformer,
                  'iTransformer': iTransformer_single, 'PatchTST': PatchTST, 'TimeXer': TimeXer, 'Proposed': Proposed}
    model_name_list = ['TCN', 'FreTS', 'LSTM', 'GRU', 'Pyraformer', 'PatchTST', 'iTransformer', 'TimeXer', 'Proposed']

    results = []
    # for site, dataset in zip(['site_1B', 'site_24', 'site_PVOD', 'site_PVOD'],
    #                          ['2021_3_2021_5_5min', '2021_3_2021_5_5min', '01_15min', '03_15min']):
    for site, dataset in zip(['site_1B', 'site_24', ],
                                 ['2021_3_2021_5_5min', ]):
        # for site, dataset in zip(['site_PVOD','site_PVOD'], ['01_15min','03_15min',]):
        # for site, dataset in zip(['site_1B'], ['2021_3_2021_5_5min']):
        # for site, dataset in zip(['elec'], ['2021_3_2021_5_5min']):
        # for idx in range(9):
        #

        for idx in [8]:
            seeds = 42
            site = site
            dataset = dataset
            # site = 'USA'
            # site = 'Australia'
            # dataset = '2018'
            parser = argparse.ArgumentParser(description="Hyperparameters")
            parser.add_argument("--batch_size", type=int, default=300)
            parser.add_argument("--learning_rate", type=float, default=0.001)
            parser.add_argument("--epochs", type=int, default=100)
            # parser.add_argument('--data_dir', type=str, default='./dataset', help='数据集的路径')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            args = parser.parse_args()
            batch_size = args.batch_size
            learning_rate = args.learning_rate
            epochs = args.epochs
            file_path = f'./data/{site}/{site}_{dataset}.csv'
            seq_len = 24 * 3
            # predict_length = [1,4]
            pred_len = 6 if '_5min' in dataset else 4
            enc_in = 5
            device = torch.device('cuda:0')
            df_all = pd.read_csv(file_path, header=0)
            multi_steps = True
            # need_train = True
            need_train = False  # 'CA_'+
            model_name = 'CA_'+model_name_list[idx] + '_' + site + '_' + dataset
            model_select = model_name_list[idx]

            data_train, data_valid, data_test, _, _, _, scalar = split_data(df_all, 0.8, 0.1, seq_len)
            dataset_train = data_detime(data=data_train, lookback_length=seq_len, multi_steps=multi_steps,
                                        lookforward_length=pred_len)
            dataset_valid = data_detime(data=data_valid, lookback_length=seq_len, multi_steps=multi_steps,
                                        lookforward_length=pred_len)
            dataset_test = data_detime(data=data_test, lookback_length=seq_len, multi_steps=multi_steps,
                                       lookforward_length=pred_len)

            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                      worker_init_fn=lambda _: same_seeds(seeds))
            valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,
                                      worker_init_fn=lambda _: same_seeds(seeds))
            test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                     worker_init_fn=lambda _: same_seeds(seeds))


            def objective(trial):

                same_seeds(seeds)

                # dim_embed = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256, 512])
                # layer_ = trial.suggest_categorical('layer_I', [1, 2, 3, 4, 5, 6])
                # heads = trial.suggest_categorical('heads', [2, 4, 6, 8, 12])
                if not need_train:
                    params = select_hyperparameters(trial, model_name=model_select, type='optimal', seq_len=seq_len,
                                                    pred_len=pred_len, enc_in=enc_in, dataset=dataset, site=site)
                else:
                    params = select_hyperparameters(trial, model_name=model_select, type='optuna', seq_len=seq_len,
                                                    pred_len=pred_len, enc_in=enc_in, dataset=dataset, site=site)
                    # params = select_hyperparameters(trial, model_name=model_select, type='optimal', seq_len=seq_len,
                    #                                 pred_len=pred_len, enc_in=enc_in, dataset=dataset, site=site)
                    # params = select_hyperparameters(trial, model_name=model_select, type='single', seq_len=seq_len,
                    #                                 pred_len=pred_len, enc_in=enc_in,dataset=dataset,site=site)
                model = model_list[model_select](**params).to(device)
                # Criterion = ql_loss
                Criterion = nn.MSELoss()
                # Criterion = nn.L1Loss(reduction='sum')
                optm = optim.Adam(model.parameters(), lr=learning_rate)
                optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optm, mode="min", factor=0.5, patience=5, verbose=True)
                model_save = f"model_save/{site}/{dataset}/{model_name}.pt"
                train_losses, valid_losses = [], []
                earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)
                if not os.path.exists(f"model_save/{site}/{dataset}"):
                    os.makedirs(f"model_save/{site}/{dataset}")
                    os.makedirs(f"model_save/{site}/{dataset}/best")
                model_save = f"model_save/{site}/{dataset}/{model_name}.pt" if need_train else f"model_save/{site}/{dataset}/best/{model_name}.pt"
                if need_train:
                    try:
                        for epoch in range(epochs):
                            time_start = time.time()
                            train_loss = train(data=train_loader, model=model, criterion=Criterion, optm=optm, )
                            valid_loss, ms = evaluate(data=valid_loader, model=model, criterion=Criterion, )
                            train_losses.append(train_loss)
                            valid_losses.append(valid_loss)
                            optm_schedule.step(valid_loss)
                            earlystopping(valid_loss, model, params)
                            # torch.save(model, model_save, pickle_module=dill)
                            print('')
                            print(
                                f'Epoch:{epoch + 1}| {model_name}|time:{(time.time() - time_start):.2f}|Loss_train:{train_loss:.4f}|Learning_rate:{optm.state_dict()["param_groups"][0]["lr"]:.4f}\n'
                                f'Loss_valid:{valid_loss:.4f}|{ms}',
                                flush=True, )
                            if earlystopping.early_stop:
                                print("Early stopping")
                                break  # 跳出迭代，结束训练
                    except KeyboardInterrupt:
                        print("Training interrupted by user")
                    # plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
                    # plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid rmse")----------------*
                    # plt.legend()  # 显示图例
                    # plt.xlabel("epoches")
                    # # plt.ylabel("epoch")
                    # plt.title("Train_loss&Valid_loss")
                    # plt.show()
                with open(model_save, "rb") as f:
                    # model = torch.load(f, pickle_module=dill)
                    checkpoint = torch.load(f)  # 只读取一次文件
                    # 尝试加载模型权重
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # 如果没有 model_state_dict 键，使用默认方式加载
                        model.load_state_dict(checkpoint)
                # print(model)
                model = model.to(device)

                for x ,y in test_loader:
                # x,_=test_loader[0]
                    visualize_feature_map(model, x,'gate')
                    break

                test_loss, ms_test, tmp = evaluate(
                    data=test_loader, model=model, criterion=Criterion, scalar=scalar)
                results.append(tmp)
                ms_test['model_name'] = model_name
                # save_metrics_to_excel(ms_test, f'data_record/multi_step.xlsx', sheet_name=f'{site}_{dataset}')
                # path_record = 'data_record/' + site + '/' + dataset + '.xlsx'
                # save_dict_to_excel(ms_test, path_record, sheet_name=model_select)
                print(
                    f'params:{params}\nTest_loss:{test_loss:.4f}| {ms_test}')
                # with open(f'data_record/{site}/{dataset}/Metrics_{model_name}.json', 'a', newline='') as f:
                #     json.dump(ms_test, f, indent=4)
                return None if not need_train else valid_loss


            #

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seeds),
                                        load_if_exists=True,
                                        storage=f'sqlite:///data_record/db_solar.sqlite3',
                                        study_name=f'{model_name}')
            study.optimize(objective, n_trials=1)

            print(study.best_params, '\n', study.best_value)

    curve_colors = ['blue', 'red']
    line_widths = [1.7, 1.7]
    # 创建 4x1 的子图
    fig, axs = plt.subplots(4, 1, figsize=(16, 12))
    nums_start=600
    nums_end = 1200

    # 绘图及设置刻度（sticks）
    ksize=16
    dataset_en=['A','B','C','D']
    ### results plot
    # for i, ax in enumerate(axs):
    #     x = np.arange(nums_end-nums_start)
    #     preds = results[i][0][nums_start:nums_end, 0]
    #     labels = results[i][1][nums_start:nums_end, 0]
    #     #
    #     # ax.plot(x, labels, color=curve_colors[1], linewidth=line_widths[1], label='真实值',alpha=0.8,marker='o',markersize=3, markeredgecolor='black',)
    #     # ax.plot(x, preds, color=curve_colors[0], linewidth=line_widths[0], label='预测值',alpha=0.8,marker='s',markersize=3, markeredgecolor='black',)
    #     ax.plot(x, labels, color=curve_colors[1], linewidth=line_widths[1], label='Observed',alpha=0.8)
    #     ax.plot(x, preds, color=curve_colors[0], linewidth=line_widths[0], label='Forecast',alpha=0.8)
    #     ax.fill_between(x,labels, preds, color='gray', alpha=0.8,label='Error Band')
    #     # 设置 x 轴的刻度为0到10，每隔1个单位
    #     # ax.set_xticks(np.arange(0, 11, 1))
    #     time_sample='5min' if i<2 else '15min'
    #     ax.set_xlabel(f'Time ({time_sample})',fontsize=ksize)
    #     ax.set_ylabel('PV Power (KW)',fontsize=ksize)
    #     ax.tick_params(axis='x', labelsize=ksize)
    #     ax.tick_params(axis='y', labelsize=ksize)
    #     if i==1:
    #         ax.set_yticks([0, 3])
    #     # 添加图例和网b格
    #     # ax.legend()
    #     # ax.grid(True)
    #     # 添加标题说明每个子图（可选）
    #     ax.set_title(f'Dataset {dataset_en[i]}',fontsize=ksize)

    # 设置整体布局，避免子图之间重叠
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='upper center', fontsize=ksize, bbox_to_anchor=(0.53, 1.0))
    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.92, bottom=0.05, left=0.1, right=0.97)  # 调整图例的位置
    # , loc='upper right', bbox_to_anchor=(1.2, 1)
    plt.savefig('pic/results_chapt3_en.svg', format='SVG', dpi=800)
    plt.show()
# optuna-dashboard sqlite:///data_record/db_solar.sqlite3
