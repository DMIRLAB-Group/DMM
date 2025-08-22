import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import os
from einops import rearrange
import torch.distributions as D
from torch.func import jacfwd, vmap
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        configs,
        dropout=0.1,
        attn_dropout=0.1
    ):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = configs.diagonal_attention_mask
        self.device = configs.gpu
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask == 1:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class MLP1(nn.Module):
    def __init__(self, layer_nums, in_dim, hid_dim=None, out_dim=None, activation="gelu", layer_norm=True):
        super().__init__()
        if activation == "gelu":
            a_f = nn.GELU()
        elif activation == "relu":
            a_f = nn.ReLU()
        elif activation == "tanh":
            a_f = nn.Tanh()
        else:
            a_f = nn.Identity()
        if out_dim is None:
            out_dim = in_dim
        if layer_nums == 1:
            net = [nn.Linear(in_dim, out_dim)]
        else:

            net = [nn.Linear(in_dim, hid_dim), a_f, nn.LayerNorm(hid_dim)] if layer_norm else [
                nn.Linear(in_dim, hid_dim), a_f]
            for i in range(layer_nums - 2):
                net.append(nn.Linear(in_dim, hid_dim))
                net.append(a_f)
            net.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MLP2(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

        
class Embedding_Net(nn.Module):

    def __init__(self, patch_size, input_len, out_len, emb_dim) -> None:
        super().__init__()
        self.patch_size = patch_size if patch_size <= input_len else input_len
        self.stride = self.patch_size // 2
        self.out_len = out_len

        self.num_patches = int((input_len - self.patch_size) / self.stride + 1)

        self.net1 = MLP1(1, in_dim=self.patch_size, out_dim=emb_dim)
        self.net2 = MLP1(1, emb_dim * self.num_patches, out_dim=self.out_len)

    def forward(self, x):
        B, L, M = x.shape
        if self.num_patches != 1:
            x = rearrange(x, 'b l m -> b m l')
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x = rearrange(x, 'b m n p -> (b m) n p')
        else:
            x = rearrange(x, 'b l m -> (b m) 1 l')
        x = self.net1(x)
        outputs = self.net2(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b  l m', b=B)
        return outputs


class ZTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, num_layers=3, hidden_dim=64, compress_dim=10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        self.gs = nn.ModuleList([MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in
                                 range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP2(input_dim=lags * latent_size + 1, hidden_dim=hidden_dim,
                  output_dim=1, num_layers=num_layers) for _ in range(latent_size)])

        self.compress = nn.Linear(lags * latent_size, compress_dim)
        self.compress_dim = compress_dim

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)

        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        if x.shape[-1] > 100:
            batch_x_lags = self.compress(batch_x_lags)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                    (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)

        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class EpsilonTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            embedding_dim,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.latent_size = latent_size
        self.lags = lags
        self.gs = nn.ModuleList([MLP2(input_dim=embedding_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in range(latent_size)])
        self.fc = MLP2(input_dim=embedding_dim, hidden_dim=hidden_dim,
                       output_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x, embeddings):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
       
        batch_embeddings = embeddings[:, -length:].expand(batch_size, length, -1).reshape(batch_size * length, -1)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1:]  # (batch_size*length, x_dim)

        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            batch_inputs = torch.cat(
                (batch_embeddings, batch_x_t[:, :, i]), dim=-1)
            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian
 
 
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.backbone == 'cnn':
            self.model = DMM_cnn(configs)
        elif configs.backbone == 'attn':
            self.model = DMM_attn(configs)
            
    def forward(self, inp, mask, training=True):

        imputed_data, other_loss = self.model(inp, mask, training)

        if training:
            return imputed_data, other_loss
        else:
            return imputed_data


class DMM_cnn(nn.Module):

    def __init__(self, configs):
        super(DMM_cnn, self).__init__()
        self.configs = configs
        self.device = configs.gpu
        self.seq_len = configs.seq_len
        self.d_layers = configs.d_layers
        self.d_model = configs.d_model
        self.tau_num = configs.tau_num
        self.DMM_type = configs.DMM_type
        
        if self.DMM_type == 'MNAR': 
            assert self.seq_len % self.tau_num == 0
            self.tau = self.seq_len // self.tau_num
        else: 
            self.tau = self.seq_len

        self.feature_dim = configs.enc_in * 2 if self.DMM_type == 'MNAR' else configs.enc_in
     
        if self.DMM_type == 'MNAR':
            self.model = nn.ModuleList([self.conv1d_with_init(self.tau, self.tau, configs.kernel_size)
                                    for _ in range(configs.tau_num)])
            self.z_std = nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim, bias=True)
                                    for _ in range(configs.tau_num)])
            self.projection = nn.ModuleList([nn.Linear(self.feature_dim, configs.c_out, bias=True)
                                    for _ in range(configs.tau_num)])
        else:
            if self.d_layers == 1:
                self.model = self.conv1d_with_init(self.tau, self.tau, configs.kernel_size)
            else:
                net = [self.conv1d_with_init(self.tau, self.d_model, configs.kernel_size), nn.LayerNorm(configs.enc_in)]
                for i in range(self.d_layers - 2):
                    net.append(self.conv1d_with_init(self.d_model, self.d_model, configs.kernel_size))
                    net.append(nn.LayerNorm(configs.enc_in))
                net.append(self.conv1d_with_init(self.d_model, self.tau, configs.kernel_size))
                self.model = nn.Sequential(*net)
        
            self.z_std = nn.Linear(self.feature_dim, self.feature_dim, bias=True)
                                           
            self.net = Embedding_Net(configs.patch_size, configs.seq_len, configs.seq_len, configs.emb_dim)

            self.projection = nn.Linear(self.feature_dim, configs.c_out, bias=True)
        
        self.z_dim = self.feature_dim // 2
        self.lag = 1
        self.z_prior = ZTransitionPrior(lags=self.lag,
                                         latent_size=self.z_dim,
                                         num_layers=1,
                                         hidden_dim=8)
        self.epsilon_prior = EpsilonTransitionPrior(lags=self.lag,
                                                     latent_size=self.feature_dim-self.z_dim,
                                                     embedding_dim=configs.enc_in,
                                                     num_layers=1,
                                                     hidden_dim=8)
        self.register_buffer('z_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('z_dist_var', torch.eye(self.z_dim))
        self.register_buffer('epsilon_dist_mean', torch.zeros(self.feature_dim-self.z_dim))
        self.register_buffer('epsilon_dist_var', torch.eye(self.feature_dim-self.z_dim))

    @property
    def epsilon_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.epsilon_dist_mean, self.epsilon_dist_var)

    @property
    def z_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.z_dist_mean, self.z_dist_var)
        
    def loss_function(self, mus, logvars, z):

        batch_size, length, _ = z.shape
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(z[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        return kld_normal
        
    def z_kl_loss(self, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lag:]
        residuals, logabsdet = self.z_prior(z_est)
        log_pz_laplace = torch.sum(self.z_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                              lags_and_length - self.lag)
        kld_laplace = kld_laplace.mean()
        loss = (kld_normal + kld_laplace)
        return loss
    
    def kl_loss(self, mus, logvars, z_est, c_embedding):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Future KLD
        log_qz_laplace = log_qz
        residuals, logabsdet = self.epsilon_prior.forward(z_est, c_embedding)

        log_pz_laplace = torch.sum(self.epsilon_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                          lags_and_length)
        kld_laplace = kld_laplace.mean()
        loss = kld_laplace
        return loss
        
    def conv1d_with_init(self, in_channels, out_channels, kernel_size):
        layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(layer.weight)
        return layer
        
    def forward(self, x, mask, training):
        B, L, N = x.shape
        x_enc = x.masked_fill(mask == 0, 0)
        means = torch.sum(x_enc, dim=1) / (torch.sum(mask == 1, dim=1)+ 1e-5)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           (torch.sum(mask == 1, dim=1)+1e-5) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        x_enc = x_enc.masked_fill(mask == 0, 0)
        
        if self.DMM_type == 'MNAR':
            z_mean_list = []
            z_std_list = []
            z_list = []
            dec_out_list = []
            dec_out = torch.randn(B, self.tau, N).to(self.device)
            for i in range(self.tau_num):
                enc_impute = torch.cat([x_enc[:, i*self.tau:(i+1)*self.tau,:], dec_out], dim=2)
                z_mean = self.model[i](enc_impute)
                z_mean_list.append(z_mean)
                z_std = self.z_std[i](enc_impute)
                z_std_list.append(z_std)
                z = self.reparametrize(z_mean, z_std) if training else z_mean
                z_list.append(z)
                dec_out = self.projection[i](z)
                dec_out_list.append(dec_out)
            z_mean = torch.cat(z_mean_list, dim=1)
            z_std = torch.cat(z_std_list, dim=1)
            z = torch.cat(z_list, dim=1)
            dec_out = torch.cat(dec_out_list, dim=1)
        else:
            z_mean = self.model(x_enc)
            x_enc = self.net(x_enc)
            z_std = self.z_std(x_enc)
            z = self.reparametrize(z_mean, z_std) if training else z_mean
            dec_out = self.projection(z)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        dec_out = mask * x + (1 - mask) * dec_out
        if training:
            if self.DMM_type == 'MAR':
                z_kld = self.z_kl_loss(z_mean[:, :, :self.z_dim], z_std[:, :, :self.z_dim], z[:, :, :self.z_dim])
                kld = self.kl_loss(z_mean[:, :, self.z_dim:], z_std[:, :, self.z_dim:], z[:, :, self.z_dim:], x)
                other_loss = self.configs.kld_weight * (z_kld + kld)
            elif self.DMM_type == 'MNAR':
                z_kld = self.z_kl_loss(z_mean[:, :, :self.z_dim], z_std[:, :, :self.z_dim], z[:, :, :self.z_dim])
                kld = self.kl_loss(z_mean[:, :, self.z_dim:], z_std[:, :, self.z_dim:], z[:, :, self.z_dim:], dec_out)
                other_loss = self.configs.kld_weight * (z_kld + kld)
        else:
            other_loss = 0
            
        return dec_out, other_loss

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

     
class DMM_attn(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.tau_num = configs.tau_num
        self.seq_len = configs.seq_len
        self.DMM_type = configs.DMM_type
        if self.DMM_type == 'MNAR': 
            assert self.seq_len % self.tau_num == 0
            self.tau = self.seq_len // self.tau_num
        else: 
            self.tau = self.seq_len
        self.n_groups = configs.n_groups
        self.n_group_inner_layers = configs.n_group_inner_layers
        self.input_with_mask = configs.input_with_mask
        self.DMM_type = configs.DMM_type
        actual_d_feature = configs.enc_in * 2  if self.input_with_mask == 1 else configs.enc_in
        encoder_d_feature = actual_d_feature + configs.enc_in  if self.DMM_type == 'MNAR' else actual_d_feature
        self.param_sharing_strategy = configs.param_sharing_strategy
        self.MIT = configs.MIT
        self.device = configs.gpu
        if configs.param_sharing_strategy == "between_group":
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.tau,
                        encoder_d_feature,
                        configs.d_model,
                        configs.d_inner,
                        configs.n_heads,
                        configs.d_k,
                        configs.d_v,
                        configs,
                        configs.dropout,
                        0,

                    )
                    for _ in range(configs.n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.tau,
                        encoder_d_feature,
                        configs.d_model,
                        configs.d_inner,
                        configs.n_heads,
                        configs.d_k,
                        configs.d_v,
                        configs,
                        configs.dropout,
                        0,

                    )
                    for _ in range(configs.n_group_inner_layers)
                ]
            )
        else:
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.tau,
                        encoder_d_feature,
                        configs.d_model,
                        configs.d_inner,
                        configs.n_heads,
                        configs.d_k,
                        configs.d_v,
                        configs,
                        configs.dropout,
                        0,
                    )
                    for _ in range(configs.n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        self.tau,
                        encoder_d_feature,
                        configs.d_model,
                        configs.d_inner,
                        configs.n_heads,
                        configs.d_k,
                        configs.d_v,
                        configs,
                        configs.dropout,
                        0,

                    )
                    for _ in range(configs.n_groups)
                ]
            )
        self.Z_dim = configs.Z_dim
        self.z_dim = configs.z_dim
        assert self.Z_dim > self.z_dim
        self.dropout = nn.Dropout(p=configs.dropout)
        self.position_enc = PositionalEncoding(configs.d_model, n_position=self.tau)
        # for the 1st block
        self.embedding_1 = nn.Linear(encoder_d_feature, configs.d_model)
        self.reduce_dim_z = nn.Linear(configs.d_model, configs.enc_in)
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_d_feature, configs.d_model)
        self.reduce_dim_beta = nn.Linear(self.Z_dim, configs.enc_in)
        self.reduce_dim_gamma = nn.Linear(configs.enc_in, configs.enc_in)
        # for the 3rd block
        self.weight_combine = nn.Linear(configs.enc_in + self.tau, configs.enc_in)

        self.z_mean = nn.Linear(configs.d_model, configs.Z_dim)
        self.z_std = nn.Sequential(nn.Linear(configs.d_model, configs.Z_dim), nn.Sigmoid())

        self.lag = 1
        self.z_transition_prior = ZTransitionPrior(lags=self.lag,
                                                             latent_size=self.z_dim,
                                                             num_layers=1,
                                                             hidden_dim=8)
        self.epsilon_transition_prior = EpsilonTransitionPrior(lags=self.lag,
                                                              latent_size=self.Z_dim-self.z_dim,
                                                              embedding_dim=configs.enc_in,
                                                              num_layers=1,
                                                              hidden_dim=8)
        self.register_buffer('z_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('z_dist_var', torch.eye(self.z_dim))
        self.register_buffer('epsilon_dist_mean', torch.zeros(self.Z_dim-self.z_dim))
        self.register_buffer('epsilon_dist_var', torch.eye(self.Z_dim-self.z_dim))

    @property
    def epsilon_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.epsilon_dist_mean, self.epsilon_dist_var)

    @property
    def z_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.z_dist_mean, self.z_dist_var)
        
    def loss_function(self, mus, logvars, z):

        batch_size, length, _ = z.shape
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(z[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        
        return kld_normal
        
    def z_kl_loss(self, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lag:]
        residuals, logabsdet = self.z_transition_prior(z_est)
        log_pz_laplace = torch.sum(self.z_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                              lags_and_length - self.lag)
        kld_laplace = kld_laplace.mean()
        loss = (kld_normal + kld_laplace)
        return loss
    
    def kl_loss(self, mus, logvars, z_est, c_embedding):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Future KLD
        log_qz_laplace = log_qz
        residuals, logabsdet = self.epsilon_transition_prior.forward(z_est, c_embedding)

        log_pz_laplace = torch.sum(self.epsilon_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                          lags_and_length)
        kld_laplace = kld_laplace.mean()
        loss = kld_laplace
        return loss

    def impute(self, X, masks, training, imputed_data=None):
    
        input_X_for_first = torch.cat([X, imputed_data], dim=2) if self.DMM_type == 'MNAR' else X
        input_X_for_first = torch.cat([input_X_for_first, masks], dim=2) if self.input_with_mask == 1 else input_X_for_first
        input_X_for_first = self.embedding_1(input_X_for_first.float())
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        
        X_prime = masks * X + (1 - masks) * X_tilde_1
        
        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask == 1 else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second.float())
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        z_mean, z_std = self.z_mean(enc_output), self.z_std(enc_output)
        z = self.__reparametrize(z_mean, z_std) if training else z_mean

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(z)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)
        
        attn_weights = torch.cat([masks, attn_weights], dim=2)
        combining_weights = self.weight_combine(attn_weights.float())
        combining_weights = F.sigmoid(combining_weights)# namely term eta
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data

        X_c = masks * X + (1 - masks) * X_tilde_3

        return X_c, z_mean, z_std, z 

    def forward(self, x, masks, training=True):
    
        if self.DMM_type == 'MNAR':
            B, L, N = x.shape
            z_mean_list = []
            z_std_list = []
            z_list = []
            imputed_data_list = []
            imputed_data = torch.randn(B, self.tau, N).to(self.device)
            for i in range(self.tau_num):
                mask = masks[:, i*self.tau:(i+1)*self.tau,:]
                imputed_data, z_mean, z_std, z = self.impute(x[:, i*self.tau:(i+1)*self.tau,:], mask, training, imputed_data)
                z_mean_list.append(z_mean)
                z_std_list.append(z_std)
                z_list.append(z)
                imputed_data_list.append(imputed_data)
            z_mean = torch.cat(z_mean_list, dim=1)
            z_std = torch.cat(z_std_list, dim=1)
            z = torch.cat(z_list, dim=1)
            imputed_data = torch.cat(imputed_data_list, dim=1)
        else:
            imputed_data, z_mean, z_std, z = self.impute(x, masks, training)
            
        if training:
            if self.DMM_type == 'MAR':
                z_kld = self.z_kl_loss(z_mean[:, :, :self.z_dim], z_std[:, :, :self.z_dim], z[:, :, :self.z_dim])
                kld = self.kl_loss(z_mean[:, :, self.z_dim:], z_std[:, :, self.z_dim:], z[:, :, self.z_dim:], x)
                other_loss = self.configs.kld_weight * (z_kld + kld)
            elif self.DMM_type == 'MNAR':
                z_kld = self.z_kl_loss(z_mean[:, :, :self.z_dim], z_std[:, :, :self.z_dim], z[:, :, :self.z_dim])
                kld = self.kl_loss(z_mean[:, :, self.z_dim:], z_std[:, :, self.z_dim:], z[:, :, self.z_dim:], imputed_data)
                other_loss = self.configs.kld_weight * (z_kld + kld)
        else:
            other_loss = 0

        return imputed_data, other_loss

    def __reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z


