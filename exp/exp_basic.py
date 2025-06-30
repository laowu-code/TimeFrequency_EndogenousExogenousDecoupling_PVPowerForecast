import os
import torch
from models_origin import Autoformer, DLinear, TiDE, FreTS,TimeXer,iTransformer,PatchTST,LSTM,GRU,TCN,Ours
from torch import optim

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'TimeXer': TimeXer,
            'LSTM' : LSTM,
            'GRU' : GRU,
            "TCN": TCN,
            "Ours": Ours,

        }
        # if args.model_name == 'Mamba':
        #     print('Please make sure you have successfully installed mamba_ssm')
        #     from models_origin import Mamba
        #     self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.optim=optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optim_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="min", factor=0.5, patience=5, verbose=True
        )

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
