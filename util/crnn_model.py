"""
YL

The nobiko_opt.neko names are for the LSTM options:

Maru - CRNN
Haku - CRNN
Pixel - CNN only

Requires util.restnet_3d.py for the 3D resnet.

"""

import torch
import torch.nn as nn
import util.resnet_3d as tora_3dresnet


class Maru(nn.Module):
    def __init__(self, tora_opt):
        super().__init__()
        # <<< CNN PORTION >>>
        if tora_opt.model_depth == 10:
            self.resnet_3d = tora_3dresnet.resnet10()
        elif tora_opt.model_depth == 18:
            self.resnet_3d = tora_3dresnet.resnet18()
        elif tora_opt.model_depth == 34:
            self.resnet_3d = tora_3dresnet.resnet34()
        elif tora_opt.model_depth == 50:
            self.resnet_3d = tora_3dresnet.resnet50()
        elif tora_opt.model_depth == 101:
            self.resnet_3d = tora_3dresnet.resnet101()

        # <<< RNN PORTION >>>
        self.input_size = tora_opt.rnn_input  
        self.hidden_size = tora_opt.rnn_hidden
        # 4 gates
        self.W_ih = nn.Parameter(torch.Tensor(tora_opt.rnn_input, tora_opt.rnn_hidden * 4))
        self.W_hh = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden, tora_opt.rnn_hidden * 4))
        self.biav = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden * 4))

        # <<< TAILS - FCs >>>
        self.tail = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                  nn.ReLU(),
                                  nn.Linear(tora_opt.rnn_hidden, 2))

        self.init_weights()

    def init_weights(self):
        # initialise the weights for the RNN portion
        for name, param in self.named_parameters():
            if 'biav' in name:
                nn.init.constant_(param, 0.0)
            elif 'W_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'W_hh' in name:
                nn.init.orthogonal_(param)
            # TODO further investigation on the init method

    def forward(self, x, init_states=None):
        # x is 5D tensor
        bs, seq_size = x.size(0), x.size(1)
        # print('bath_s', bs)
        # print('seq', seq_size)

        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
        else:
            h_t, c_t = init_states

        # CNN-RNN Recurrent calculation
        for t in range(seq_size):
            x_t = x[:, t, :, :, :]
            x_t = x_t.unsqueeze(1)  

            # x_t (batch_size, feature = LSTM input size)

            x_t = self.resnet_3d(x_t)
            gates = x_t @ self.W_ih + h_t @ self.W_hh + self.biav

            # forget_gate_t, input_gate_t, output_gate_t, C~ candidate/ gate_gate
            f_t, i_t, o_t, c_tmp = torch.chunk(gates, 4, 1)
            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(c_tmp)  

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        x = self.tail(h_t)

        return x


class Maru_L0(nn.Module):
    # L0 version of the MARU lstm, this uses one of the LCs in Pixel as cnn_lc
    # This forces the RNN_IN to be 32
    def __init__(self, tora_opt):
        super().__init__()

        # special mode to adjust the shape into the saliencymap_mode compatible one
        self.saliencymap_mode = tora_opt.saliencymap_mode

        # <<< CNN PORTION >>>
        if tora_opt.model_depth == 10:
            self.resnet_3d = tora_3dresnet.resnet10()
        elif tora_opt.model_depth == 18:
            self.resnet_3d = tora_3dresnet.resnet18()
        elif tora_opt.model_depth == 34:
            self.resnet_3d = tora_3dresnet.resnet34()
        elif tora_opt.model_depth == 50:
            self.resnet_3d = tora_3dresnet.resnet50()
        elif tora_opt.model_depth == 101:
            self.resnet_3d = tora_3dresnet.resnet101()

        # <<< RNN PORTION >>>
        self.input_size = tora_opt.rnn_input  
        self.hidden_size = tora_opt.rnn_hidden
        # 4 gates
        self.W_ih = nn.Parameter(torch.Tensor(tora_opt.rnn_input, tora_opt.rnn_hidden * 4))
        self.W_hh = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden, tora_opt.rnn_hidden * 4))
        self.biav = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden * 4))

        # <<< L0 cnn_lc >>>
        self.cnn_lc = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                      nn.ReLU(),
                                      nn.Linear(512, 32))

        # <<< TAILS - FCs >>>
        self.tail = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                  nn.ReLU(),
                                  nn.Linear(tora_opt.rnn_hidden, 2))

        self.init_weights()

    def init_weights(self):
        # initialise the weights for the RNN portion
        for name, param in self.named_parameters():
            if 'biav' in name:
                nn.init.constant_(param, 0.0)
            elif 'W_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'W_hh' in name:
                nn.init.orthogonal_(param)
            # TODO further investigation on the init method

    def forward(self, x, init_states=None):

        bs, seq_size = x.size(0), x.size(1)

        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
        else:
            h_t, c_t = init_states

        if self.saliencymap_mode:
                x = x.unsqueeze(0)
                seq_size = 1 

        # CNN-RNN Recurrent calculation
        for t in range(seq_size):
            x_t = x[:, t, :, :, :]
            x_t = x_t.unsqueeze(1) 

            x_t = self.resnet_3d(x_t)
            x_t = self.cnn_lc(x_t)  # L0 implementation

            gates = x_t @ self.W_ih + h_t @ self.W_hh + self.biav

            # forget_gate_t, input_gate_t, output_gate_t, C~ candidate/ gate_gate
            f_t, i_t, o_t, c_tmp = torch.chunk(gates, 4, 1)
            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(c_tmp)  # gate-gate

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        x = self.tail(h_t)

        return x


class Maru_L0_cox(nn.Module):
    # cox regression version of Maru. Reduce the output from the two classes to 1
    def __init__(self, tora_opt):
        super().__init__()

        self.saliencymap_mode = tora_opt.saliencymap_mode

        # <<< CNN PORTION >>>
        if tora_opt.model_depth == 10:
            self.resnet_3d = tora_3dresnet.resnet10()
        elif tora_opt.model_depth == 18:
            self.resnet_3d = tora_3dresnet.resnet18()
        elif tora_opt.model_depth == 34:
            self.resnet_3d = tora_3dresnet.resnet34()
        elif tora_opt.model_depth == 50:
            self.resnet_3d = tora_3dresnet.resnet50()
        elif tora_opt.model_depth == 101:
            self.resnet_3d = tora_3dresnet.resnet101()

        # <<< RNN PORTION >>>
        self.input_size = tora_opt.rnn_input  
        self.hidden_size = tora_opt.rnn_hidden
        # 4 gates
        self.W_ih = nn.Parameter(torch.Tensor(tora_opt.rnn_input, tora_opt.rnn_hidden * 4))
        self.W_hh = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden, tora_opt.rnn_hidden * 4))
        self.biav = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden * 4))

        # <<< L0 cnn_lc >>>
        self.cnn_lc = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                      nn.ReLU(),
                                      nn.Linear(512, 32))

        # <<< TAILS - FCs >>>
        self.tail = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                  nn.ReLU(),
                                  nn.Linear(tora_opt.rnn_hidden, 1))

        self.init_weights()

    def init_weights(self):
        # initialise the weights for the RNN portion
        for name, param in self.named_parameters():
            if 'biav' in name:
                nn.init.constant_(param, 0.0)
            elif 'W_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'W_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, init_states=None):
        bs, seq_size = x.size(0), x.size(1)

        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
        else:
            h_t, c_t = init_states


        if self.saliencymap_mode:
                x = x.unsqueeze(0)  
                seq_size = 1 

        # CNN-RNN Recurrent calculation
        for t in range(seq_size):

            x_t = x[:, t, :, :, :]
            x_t = x_t.unsqueeze(1)  

            x_t = self.resnet_3d(x_t)
            x_t = self.cnn_lc(x_t) 

            # vectorised LSTM matrix multiplication for the 4 gates
            gates = x_t @ self.W_ih + h_t @ self.W_hh + self.biav

            # forget_gate_t, input_gate_t, output_gate_t, C~ candidate/ gate_gate
            f_t, i_t, o_t, c_tmp = torch.chunk(gates, 4, 1)
            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(c_tmp)  # gate-gate

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        x = self.tail(h_t)

        return x


class Haku(nn.Module):
    # HAKU - time aware LSTM
    # Baytas et al 2017, patient subtying via time-aware LSTM networks
    # additional gate (W_t) to get the adjusted cell state
    # TODO - https://github.com/duskybomb/tlstm/blob/master/tlstm.py

    def __init__(self, tora_opt):
        super().__init__()
        # <<< CNN PORTION >>>
        if tora_opt.model_depth == 10:
            self.resnet_3d = tora_3dresnet.resnet10()
        elif tora_opt.model_depth == 18:
            self.resnet_3d = tora_3dresnet.resnet18()
        elif tora_opt.model_depth == 34:
            self.resnet_3d = tora_3dresnet.resnet34()
        elif tora_opt.model_depth == 50:
            self.resnet_3d = tora_3dresnet.resnet50()
        elif tora_opt.model_depth == 101:
            self.resnet_3d = tora_3dresnet.resnet101()

        # <<< RNN PORTION >>>
        self.input_size = tora_opt.rnn_input  
        self.hidden_size = tora_opt.rnn_hidden
        # 4 gates
        self.W_ih = nn.Parameter(torch.Tensor(tora_opt.rnn_input, tora_opt.rnn_hidden * 4))
        self.W_hh = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden, tora_opt.rnn_hidden * 4))
        self.biav = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden * 4))

        # TA-LSTM
        # Combined weightes and biases for the cell state related gate
        self.W_t = nn.Linear(tora_opt.rnn_hidden, tora_opt.rnn_hidden)

        # <<< TAILS - FCs >>>
        self.tail = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                  nn.ReLU(),
                                  nn.Linear(tora_opt.rnn_hidden, 2))

        self.init_weights()

    def init_weights(self):
        # initialise the weights for the RNN portion
        for name, param in self.named_parameters():
            if 'biav' in name:
                nn.init.constant_(param, 0.0)
            elif 'W_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'W_hh' in name:
                nn.init.orthogonal_(param)
            elif 'W_t' in name:
                # TA-LSTM additional gate
                nn.init.uniform_(param, a=0.0, b=1.0)

    def forward(self, x, t_intervals, init_states=None):
        bs, seq_size = x.size(0), x.size(1)

        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
        else:
            h_t, c_t = init_states

        # CNN-RNN Recurrent calculation
        for t in range(seq_size):
            # x's shape is (batch_size, sequence, feature)
            x_t = x[:, t, :, :, :]
            x_t = x_t.unsqueeze(1)  

            x_t = self.resnet_3d(x_t)

            # TA-LSTM TODO
            c_s1 = torch.tanh(self.W_t(c_t)) 
            c_s2 = c_s1 * t_intervals[:, t:t + 1].expand_as(c_s1)  
            c_l = c_t - c_s1  # long-term memory
            c_adj = c_l + c_s2  # adjusted cell state

            # vectorised LSTM matrix multiplication for the 4 gates
            gates = x_t @ self.W_ih + h_t @ self.W_hh + self.biav
 
            # forget_gate_t, input_gate_t, output_gate_t, C~ candidate/ gate_gate
            f_t, i_t, o_t, c_tmp = torch.chunk(gates, 4, 1)
            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(c_tmp)  # gate-gate

            c_t = f_t * c_adj + i_t * g_t  # TA-LSTM, here the c_t is replaced with c_adj
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
            # print(t, h_t)

        # print(hidden_seq)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        x = self.tail(h_t)

        return x


class Haku_L0(nn.Module):
    # HAKU - time aware LSTM
    # Baytas et al 2017, patient subtying via time-aware LSTM networks
    # additional gate (W_t) to get the adjusted cell state
    # TODO - https://github.com/duskybomb/tlstm/blob/master/tlstm.py

    def __init__(self, tora_opt):
        super().__init__()
        # <<< CNN PORTION >>>
        if tora_opt.model_depth == 10:
            self.resnet_3d = tora_3dresnet.resnet10()
        elif tora_opt.model_depth == 18:
            self.resnet_3d = tora_3dresnet.resnet18()
        elif tora_opt.model_depth == 34:
            self.resnet_3d = tora_3dresnet.resnet34()
        elif tora_opt.model_depth == 50:
            self.resnet_3d = tora_3dresnet.resnet50()
        elif tora_opt.model_depth == 101:
            self.resnet_3d = tora_3dresnet.resnet101()

        # <<< RNN PORTION >>>
        self.input_size = tora_opt.rnn_input 
        self.hidden_size = tora_opt.rnn_hidden
        # 4 gates
        self.W_ih = nn.Parameter(torch.Tensor(tora_opt.rnn_input, tora_opt.rnn_hidden * 4))
        self.W_hh = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden, tora_opt.rnn_hidden * 4))
        self.biav = nn.Parameter(torch.Tensor(tora_opt.rnn_hidden * 4))

        # <<< L0 cnn_lc >>>
        # Note the layer shape is fixed by the pretrained CNN weights
        self.cnn_lc = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                        nn.ReLU(),
                                        nn.Linear(512, 32))

        # TA-LSTM
        # Combined weightes and biases for the cell state related gate
        self.W_t = nn.Linear(tora_opt.rnn_hidden, tora_opt.rnn_hidden)

        # <<< TAILS - FCs >>>
        self.tail = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                  nn.ReLU(),
                                  nn.Linear(tora_opt.rnn_hidden, 2))

        self.init_weights()

    def init_weights(self):
        # initialise the weights for the RNN portion
        for name, param in self.named_parameters():
            if 'biav' in name:
                nn.init.constant_(param, 0.0)
            elif 'W_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'W_hh' in name:
                nn.init.orthogonal_(param)
            elif 'W_t' in name:
                # TA-LSTM additional gate
                nn.init.uniform_(param, a=0.0, b=1.0)

    def forward(self, x, t_intervals, init_states=None):

        bs, seq_size = x.size(0), x.size(1)

        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size, requires_grad=False).to(x.device)
        else:
            h_t, c_t = init_states

        # CNN-RNN Recurrent calculation
        for t in range(seq_size):
            x_t = x[:, t, :, :, :]
            x_t = x_t.unsqueeze(1)  

            x_t = self.resnet_3d(x_t)
            x_t = self.cnn_lc(x_t)  # L0 implementation

            # TA-LSTM TODO
            c_s1 = torch.tanh(self.W_t(c_t))  # short-term memory
            # here the c_t is actually C_(t-1) from the previous sequence
            c_s2 = c_s1 * t_intervals[:, t:t + 1].expand_as(c_s1)  # t_intervals, output from the G function
            c_l = c_t - c_s1  # long-term memory
            c_adj = c_l + c_s2  # adjusted cell state

            # vectorised LSTM matrix multiplication for the 4 gates
            gates = x_t @ self.W_ih + h_t @ self.W_hh + self.biav

            # forget_gate_t, input_gate_t, output_gate_t, C~ candidate/ gate_gate
            f_t, i_t, o_t, c_tmp = torch.chunk(gates, 4, 1)
            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            o_t = torch.sigmoid(o_t)
            g_t = torch.tanh(c_tmp)  # gate-gate

            c_t = f_t * c_adj + i_t * g_t  # TA-LSTM, here the c_t is replaced with c_adj
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        x = self.tail(h_t)

        return x


class Pixel(nn.Module):
    # Maru - merging tora_3dresnet with ChatoraLSTMs
    def __init__(self, tora_opt):
        super().__init__()
        # <<< CNN PORTION >>>
        if tora_opt.model_depth == 10:
            self.resnet_3d = tora_3dresnet.resnet10()
            block_expansion = 1
        elif tora_opt.model_depth == 18:
            self.resnet_3d = tora_3dresnet.resnet18()
            block_expansion = 1
        elif tora_opt.model_depth == 34:
            self.resnet_3d = tora_3dresnet.resnet34()
            block_expansion = 1
        elif tora_opt.model_depth == 50:
            self.resnet_3d = tora_3dresnet.resnet50()
            block_expansion = 4
        elif tora_opt.model_depth == 101:
            self.resnet_3d = tora_3dresnet.resnet101()
            block_expansion = 4

        # <<< NLST DumbNN-076 model 512-32-2 >>>
        self.first_lc = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                      nn.ReLU(),
                                      nn.Linear(512 * block_expansion * 1, 32))
        self.nlst_cat = nn.Sequential(nn.Dropout(tora_opt.dropout_prob),
                                      nn.ReLU(),
                                      nn.Linear(32, 2))

    def forward(self, x):
        # x is 5D tensor
        bs, seq_size = x.size(0), x.size(1)

        x_t = x[:, -1, :, :, :]
        x_t = x_t.unsqueeze(1)  
        x_t = self.resnet_3d(x_t)
        x = self.first_lc(x_t)
        x = self.nlst_cat(x)

        return x

def prepare_model(model, opt):

    if not opt.no_cuda:
        model = nn.DataParallel(model)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    if opt.phase != 'rebuild' and opt.pretrain_path:
        print('-' * 75, flush=True)
        print('<< Loading pretrained model {} >>'.format(opt.pretrain_path), flush=True)
        print('-' * 75, flush=True)
        print()
        pretrain = torch.load(opt.pretrain_path)

        pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}

        # print(pretrain['state_dict'].items())

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            # print(pname)
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    print('New_parameter: ', pname, p.shape, flush=True)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 'new_parameters': new_parameters}
        print()
        return model, parameters

    return model, model.parameters()
