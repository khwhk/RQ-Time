import torch
import torch.nn as nn
import torch.distributions as D
from einops import rearrange, repeat
from vqshape.networks import MLP, ShapeDecoder


# Utility functions
def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

def onehot_straight_through(p: torch.Tensor):
    max_idx = p.argmax(-1)
    onehot = nn.functional.one_hot(max_idx, p.shape[-1])
    return onehot + p - p.detach()

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapeDecoderMLP(nn.Module):
    """使用MLP将token解码为时间序列片段"""
    
    def __init__(self, dim_code, patch_size, hidden_dims=None, dropout=0.1):
        super().__init__()
        self.dim_code = dim_code
        self.patch_size = patch_size
        
        # 定义网络层
        if hidden_dims is None:
            hidden_dims = [dim_code * 2, dim_code * 4, dim_code * 2]
        
        layers = []
        input_dim = dim_code
        
        # 构建MLP
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, patch_size))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, z_q):
        """
        参数:
            z_q: (batch_size, num_tokens, dim_code)
        返回:
            patches: (batch_size, num_tokens, patch_size)
        """
        batch_size, num_tokens, _ = z_q.shape
        
        # 展平batch和token维度
        z_flat = z_q.reshape(-1, self.dim_code)
        
        # 通过MLP
        patches_flat = self.net(z_flat)
        
        # 恢复原始维度
        patches = patches_flat.reshape(batch_size, num_tokens, self.patch_size)
        
        return patches

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, x):
        # Create a tensor of positional indices: [0, 1, 2, ..., max_seq_length-1]
        position_indices = torch.arange(0, x.size(1)).long().unsqueeze(0).to(x.device)
        
        # Retrieve the positional embeddings corresponding to the indices
        pos_embeddings = self.positional_embeddings(position_indices)
        
        return pos_embeddings + x


class EuclCodebook(nn.Module):
    def __init__(
            self, 
            num_code: int = 512, 
            dim_code: int = 256, 
            commit_loss=1., 
            entropy_loss=0., 
            entropy_gamma=1.,
        ):
        super().__init__()
        self.num_codebook_vectors = num_code
        self.latent_dim = dim_code
        self.commit_loss = commit_loss
        self.entropy_loss = entropy_loss
        self.entropy_gamma = entropy_gamma
        
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)  # 码本：[num_code, dim_code]
        # 初始化码本向量在[-1/num_code, 1/num_code]范围内
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        # 计算输入特征z与码本向量的欧氏距离
        z_flattened = rearrange(z, "B L E -> (B L) E")  # 展平为 [B*L, E]  (bs, num_patch, dim_code) -> (bs*num_patch, dim_code)

        # Compute distance between z and codebook vectors
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))  # 距离公式优化  (bs*num_patch, num_code)

        # Find the nearest codebook vector for each z
        min_encoding_indices = torch.argmin(d, dim=1)  # 找到最近的码本索引
        # 量化后的特征
        z_q = rearrange(self.embedding(min_encoding_indices), '(B L) E -> B L E', B=z.shape[0])  # (bs, num_patch, dim_code)

        # Commitment loss
        # 承诺损失（commitment loss）：确保输入特征与量化特征接近
        loss = torch.mean((z_q.detach() - z)**2) + self.commit_loss * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        min_encoding_indices = rearrange(min_encoding_indices, '(B L) -> B L', B=z.shape[0])

        # Entropy loss
        if self.entropy_loss > 0:
            p = nn.functional.softmin(d/0.01, dim=-1)
            entropy_loss = entropy(p).mean() - self.entropy_gamma * entropy(p.mean(0))
            loss += self.entropy_loss * entropy_loss
        
        res = z - z_q  # (bs, num_patch, dim_code)

        return z_q, min_encoding_indices, loss, res  # 量化特征、码本索引、损失、残差


class PatchEncoder(nn.Module):
    def __init__(
            self,
            dim_embedding: int = 256,
            patch_size: int = 8,
            num_patch: int = 64,
            num_head: int = 6,
            num_layer: int = 6,
        ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patch = num_patch

        # Embedding Layers
        self.pos_embed = PositionalEmbedding(num_patch, dim_embedding)
        self.input_project = nn.Linear(patch_size, dim_embedding)
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer
        )

    def patch_and_embed(self, x):
        # 分割时间序列为patch：[B, num_patch, patch_size]
        x = x.unfold(-1, self.patch_size, int(x.shape[-1]/self.num_patch))
        x = self.pos_embed(self.input_project(x))
        return x

    def forward(self, x):
        # 输出编码后的patch特征
        return self.transformer(self.patch_and_embed(x))
    

class PatchDecoder(nn.Module):
    def __init__(
            self, 
            dim_embedding: int = 256,
            patch_size: int = 8,
            num_head: int = 6,
            num_layer: int = 6,
        ):
        super().__init__()

        self.patch_size = patch_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer
        )
        self.out_layer = nn.Linear(dim_embedding, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_embedding)/dim_embedding)

    def forward(self, x):
        x = torch.cat([repeat(self.cls_token, '1 1 E -> B 1 E', B=x.shape[0]), x], dim=1)  # (batch_size, num_patch+1, dim_embedding)
        out = self.transformer(x)  # (batch_size, num_patch+1, dim_embedding)
        x_hat = rearrange(self.out_layer(out[:, 1:, :]), "B L E -> B (L E)")  # (batch_size, normalize_length)
        return x_hat, out[:, 0, :]


class Tokenizer(nn.Module):
    def __init__(
            self,
            dim_embedding: int = 256,
            num_token: int = 32,
            num_head: int = 6,
            num_layer: int = 6,
        ):
        super().__init__()

        self.tokens = nn.Parameter(torch.randn(1, num_token, dim_embedding)/dim_embedding)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_embedding,
            nhead=num_head,
            dropout=0.1,
            dim_feedforward=dim_embedding*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layer
        )

    def forward(self, x, memory_mask=None):
        return self.transformer(repeat(self.tokens, '1 n d -> b n d', b=x.shape[0]), x, memory_key_padding_mask=memory_mask)


class AttributeDecoder(nn.Module):
    '''
    Decode embeddings into shape attributes
    '''
    def __init__(self, dim_code: int = 256, dim_embedding: int = 256) -> None:
        super().__init__()

        self.z_head = MLP(dim_embedding, dim_code, dim_embedding)
        # self.tl_mean_head = MLP(dim_embedding, 2, dim_embedding)
        # self.mu_head = MLP(dim_embedding, 1, dim_embedding)
        # self.sigma_head = MLP(dim_embedding, 1, dim_embedding)

    def forward(self, x):
        return self.z_head(x)

    

class AttributeEncoder(nn.Module):
    '''
    Encode shape attributes into embeddings
    '''
    def __init__(self, dim_code: int = 256, dim_embedding: int = 256) -> None:
        super().__init__()

        self.project = nn.Linear(dim_code, dim_embedding)
    
    def forward(self, z):
        return self.project(z)


def extract_subsequence(x, t, l, norm_length, smooth=9):
    '''
    Sample subsequences specified by t and l from time series x
    '''
    B, T = x.shape
    # 创建相对位置网格
    relative_positions = torch.linspace(0, 1, steps=norm_length).to(x.device)
    # 将归一化参数映射到实际索引
    start_indices = (t * (T-1))  # t ∈ [0,1] → 实际起始索引
    end_indices = (torch.clamp(t + l, max=1) * (T-1))  # 确保不超出边界
    
    # 创建采样网格：在[start, end]之间均匀采样norm_length个点
    grid = start_indices + (end_indices - start_indices) * relative_positions.unsqueeze(0)
    grid = 2.0 * grid / (T - 1) - 1
    grid = torch.stack([grid, torch.ones_like(grid)], dim=-1)
    # 双线性插值采样子序列
    x = x.unsqueeze(1).unsqueeze(2)
    interpolated = nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return moving_average(interpolated.squeeze(1).squeeze(1), smooth)


def moving_average(x, window_size):
    B, C, _ = x.shape
    filter = torch.ones(C, 1, window_size, device=x.device) / window_size
    padding = window_size // 2 # Tuple for padding (left, right)
    x = torch.cat([torch.ones(B, C, padding, device=x.device)*x[:, :, [0]], x, torch.ones(B, C, padding, device=x.device)*x[:, :, [-1]]], dim=-1)
    smoothed_x = nn.functional.conv1d(x, filter, groups=C)
    
    return smoothed_x


def eucl_sim_loss(x, threshold=0.1):
    d = torch.norm(x.unsqueeze(1) - x.unsqueeze(2), dim=-1)
    loss = nn.functional.relu(threshold - d)
    mask = torch.ones_like(loss) - torch.eye(loss.shape[-1], device=loss.device).unsqueeze(0)
    return (loss * mask).mean()

class moving_avg(nn.Module):
    """移动平均块：提取时间序列趋势项"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 两端填充，保证输出长度与输入一致
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # 适配AvgPool1d维度要求 (bs, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

# 定义序列分解模块
class series_decomp(nn.Module):
    """序列分解块：分解为残差项（高频）和趋势项（低频）"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class VQShape(nn.Module):
    def __init__(
            self, 
            dim_embedding: int = 256, # Embedding dimension of Transformers
            patch_size: int = 8, # Patch size of PatchTST backbone
            num_patch: int = 64, # Number of patches of PatchTST backbone
            num_enc_head: int = 6, # Number of heads in Transformer encoder
            num_enc_layer: int = 6, # Number of layers in Transformer encoder
            num_tokenizer_head: int = 6, # Number of heads in Transformer tokenizer
            num_tokenizer_layer: int = 6, # Number of layers in Transformer tokenizer
            num_dec_head: int = 6, # Number of heads in Transformer decoder
            num_dec_layer: int = 6, # Number of layers in Transformer decoder
            num_token: int = 32, # Number of shape tokens (output of tokenizer)
            len_s: int = 256, # Unified length of shapes
            len_input: int = 512, # Unified length of input time series
            s_smooth_factor: int = 11, # Smoothing factor for moving average
            num_code: int = 512, # Codebook size
            dim_code: int = 8, # Shape code dimension
            codebook_type: str = "standard", # Type of codebook
            lambda_commit: float = 1., # Commitment loss coefficient
            lambda_entropy: float = 1., # Entropy loss coefficient of the codebook
            entropy_gamma: float = 1., # Entropy gamma of the codebook
            mask_ratio: float = 0.25 # Mask ratio for pretraining
        ):
        super().__init__()

        self.decomp = series_decomp(kernel_size=15)
        
        self.len_s = len_s
        self.s_smooth_factor = s_smooth_factor
        self.num_code = num_code
        self.codebook_type = codebook_type
        self.min_shape_len = 1/12
        self.entropy_gamma = entropy_gamma

        self.num_patch = num_patch
        self.patch_size = patch_size  
        self.mask_ratio = mask_ratio
        self.num_token = num_token

        self.encoder = PatchEncoder(
            dim_embedding=dim_embedding,
            patch_size=patch_size,
            num_patch=num_patch,
            num_head=num_enc_head,
            num_layer=num_enc_layer
        )

        if codebook_type == "standard":
            self.codebook1 = EuclCodebook(
                num_code, 
                dim_code, 
                commit_loss=lambda_commit,
                entropy_loss=lambda_entropy,
                entropy_gamma=entropy_gamma
            )
            self.codebook2 = EuclCodebook(
                num_code * 2, 
                dim_code, 
                commit_loss=lambda_commit,
                entropy_loss=lambda_entropy,
                entropy_gamma=entropy_gamma
            )
            self.codebook3 = EuclCodebook(
                num_code * 2, 
                dim_code, 
                commit_loss=lambda_commit,
                entropy_loss=lambda_entropy,
                entropy_gamma=entropy_gamma
            )
        else:
            raise NotImplementedError(f"Invalid codebook type [{codebook_type}].")
        
        self.decoder = PatchDecoder(
            dim_embedding=dim_embedding,
            patch_size=int(len_input / num_patch),
            num_head=num_dec_head,
            num_layer=num_dec_layer
        )

        self.tokenizer = Tokenizer(
            dim_embedding=dim_embedding,
            num_token=num_patch,
            num_head=num_tokenizer_head,
            num_layer=num_tokenizer_layer
        )

        self.attr_encoder = AttributeEncoder(dim_code=dim_code, dim_embedding=dim_embedding)
        self.attr_decoder = AttributeDecoder(dim_code=dim_code, dim_embedding=dim_embedding)
        self.shape_decoder = ShapeDecoderMLP(dim_code=dim_code, patch_size=self.patch_size)

    def forward(self, x, *, mode='pretrain', num_input_patch=-1, mask=None, finetune=False):
        '''
        x: shape (batch_size, time_steps), time series data
        mode: mode of the forward pass
        num_input_patch: number of patches of the input time series (!! set if x is a partial time series, e.g. forecasting)
        mask: mask that indicates the missing values in the input time series (for imputation)
        finetune: whether to compute loss and update parameters for downstream tasks
        '''
        if mode == 'pretrain':
            return self.pretrain(x)
        elif mode == 'evaluate':
            return self.evaluate(x)
        elif mode == 'tokenize':
            return self.tokenize(x)
        elif mode == 'forecast':
            return self.forecast(x, num_input_patch, finetune)
        elif mode == 'imputation':
            return self.imputation(x, mask, finetune)
        else:
            raise NotImplementedError(f"VQShape: Invalid mode [{mode}]")
        
    def tokenize(self, x: torch.Tensor):
        self.x_mean = x.mean(dim=-1, keepdims=True)  # (batch_size, 1)
        self.x_std = (x.var(dim=-1, keepdims=True) + 1e-5).sqrt()  # (batch_size, 1)
        x_p = (x - self.x_mean) / self.x_std  # (batch_size, normalize_length)
        season_res, trend = self.decomp(x_p.unsqueeze(-1))  # 分解时间序列  x.unsqueeze(1) -> (batch_size, normalize_length, 1)
        self.season_res, self.trend = season_res.squeeze(-1), trend.squeeze(-1)  # (batch_size, normalize_length)
        self.season_res_mean = self.season_res.mean(dim=-1, keepdims=True)
        self.season_res_std = (self.season_res.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        self.trend_mean = self.trend.mean(dim=-1, keepdims=True)
        self.trend_std = (self.trend.var(dim=-1, keepdims=True) + 1e-5).sqrt()


        season_res_embed = self.encoder((self.season_res - self.season_res_mean) / self.season_res_std)  # (batch_size, num_patch, dim_embedding)
        trend_embed = self.encoder((self.trend - self.trend_mean) / self.trend_std)  # (batch_size, num_patch, dim_embedding)
        output_dict = self._forward(x, season_res_embed, trend_embed, None, compute_loss=False)

        # Token embedding
        # tokens = torch.cat([output_dict['code'], output_dict['t_pred'], output_dict['l_pred'], output_dict['mu_pred'], output_dict['sigma_pred']], dim=-1)  # (batch_size, num_token, dim_code + 4)
        tokens = torch.cat([output_dict['code_trend'], output_dict['code_season'], output_dict['code_res']], dim=-1)  # (batch_size, num_token, dim_code * 3)
        # Histogram embedding
        # histogram = torch.zeros(output_dict['code'].shape[0], self.num_code, device=x.device, dtype=output_dict['code_idx'].dtype).scatter_add_(1, output_dict['code_idx'], torch.ones_like(output_dict['code_idx']))
        histogram_trend = torch.zeros(output_dict['code_trend'].shape[0], self.num_code, device=x.device, dtype=output_dict['code_idx_trend'].dtype).scatter_add_(1, output_dict['code_idx_trend'], torch.ones_like(output_dict['code_idx_trend']))
        histogram_season = torch.zeros(output_dict['code_season'].shape[0], self.num_code * 2, device=x.device, dtype=output_dict['code_idx_season'].dtype).scatter_add_(1, output_dict['code_idx_season'], torch.ones_like(output_dict['code_idx_season']))
        histogram_res = torch.zeros(output_dict['code_res'].shape[0], self.num_code * 2, device=x.device, dtype=output_dict['code_idx_res'].dtype).scatter_add_(1, output_dict['code_idx_res'], torch.ones_like(output_dict['code_idx_res']))
        histogram = torch.cat([histogram_trend, histogram_season, histogram_res], dim=-1)   # (batch_size, num_code + num_code*2 + num_code*2)

        representations = {
            'token': tokens,
            'histogram': histogram
        }

        return representations, output_dict
    
    def evaluate(self, x: torch.Tensor):
        # print(f"[DEBUG] 输入x shape: {x.shape}")  # 应该是 (batch_size, sequence_length)
        self.x_mean = x.mean(dim=-1, keepdims=True)  # (batch_size, 1)
        self.x_std = (x.var(dim=-1, keepdims=True) + 1e-5).sqrt()  # (batch_size, 1)
        x_p = (x - self.x_mean) / self.x_std  # (batch_size, normalize_length)
        # print(f"[DEBUG] x_p (归一化后) shape: {x_p.shape}")
        season_res, trend = self.decomp(x_p.unsqueeze(-1))  # 分解时间序列  x.unsqueeze(1) -> (batch_size, normalize_length, 1)
        
        # print(f"[DEBUG] decomp输出 - season_res shape: {season_res.shape}, trend shape: {trend.shape}")
        self.season_res, self.trend = season_res.squeeze(-1), trend.squeeze(-1)  # (batch_size, normalize_length)
        # print(f"[DEBUG] squeeze后 - season_res shape: {self.season_res.shape}, trend shape: {self.trend.shape}")
        self.season_res_mean = self.season_res.mean(dim=-1, keepdims=True)
        self.season_res_std = (self.season_res.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        self.trend_mean = self.trend.mean(dim=-1, keepdims=True)
        self.trend_std = (self.trend.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        # print(f"[DEBUG] trend_mean shape: {self.trend_mean.shape}, trend_std shape: {self.trend_std.shape}")


        season_res_embed = self.encoder((self.season_res - self.season_res_mean) / self.season_res_std)  # (batch_size, num_patch, dim_embedding)
        trend_embed = self.encoder((self.trend - self.trend_mean) / self.trend_std)  # (batch_size, num_patch, dim_embedding)
    

        return self._forward(x, season_res_embed, trend_embed, None, compute_loss=True)

    def pretrain(self, x: torch.Tensor):
        self.x_mean = x.mean(dim=-1, keepdims=True)  # (batch_size, 1)
        self.x_std = (x.var(dim=-1, keepdims=True) + 1e-5).sqrt()  # (batch_size, 1)
        # Patch and embed the ts data
        x_p = (x - self.x_mean) / self.x_std  # (batch_size, normalize_length)
        season_res, trend = self.decomp(x_p.unsqueeze(-1))  # 分解时间序列  x.unsqueeze(1) -> (batch_size, normalize_length, 1)
        self.season_res, self.trend = season_res.squeeze(-1), trend.squeeze(-1)  # (batch_size, normalize_length)

        self.season_res_mean = self.season_res.mean(dim=-1, keepdims=True)
        self.season_res_std = (self.season_res.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        # season_res = (season_res - self.season_res_mean) / self.season_res_std

        self.trend_mean = self.trend.mean(dim=-1, keepdims=True)
        self.trend_std = (self.trend.var(dim=-1, keepdims=True) + 1e-5).sqrt()
        # trend = (trend - self.trend_mean) / self.trend_std

        # x_decomp = torch.stack([season_res, trend], dim=-1)  # (batch_size, normalize_length, 2)

        season_res_embed = self.encoder((self.season_res - self.season_res_mean) / self.season_res_std)  # (batch_size, num_patch, dim_embedding)
        trend_embed = self.encoder((self.trend - self.trend_mean) / self.trend_std)  # (batch_size, num_patch, dim_embedding)


        return self._forward(x, season_res_embed, trend_embed, None, compute_loss=True)

    def _forward(
            self, 
            x: torch.Tensor, 
            season_res_embed: torch.Tensor, 
            trend_embed: torch.Tensor, 
            tokenizer_attn_mask: torch.Tensor, 
            compute_loss: bool = False
        ):
        '''
        x: shape (batch_size, time_steps), time series data
        x_embed: shape (batch_size, num_patch, dim_embedding), embedded patches of the input time series
        tokenizer_attn_mask: shape (batch_size, num_token), mask that indicates the missing values in the input patches
        compute_loss: whether to compute loss
        '''

        z_e_season = self.attr_decoder(season_res_embed)  # (batch_size, num_patch, dim_code)
        z_e_trend = self.attr_decoder(trend_embed)  # (batch_size, num_patch, dim_code)

        z_q_trend, z_idx_trend, z_loss_trend, res_trend = self.codebook1(z_e_trend)  # (batch_size, num_patch, dim_code)  Trend-Codebook1
        z_q_season, z_idx_season, z_loss_season, res_season = self.codebook2(z_e_season)  # (batch_size, num_patch, dim_code)  Seasonal-Codebook2
        

        z_e_res = res_season + res_trend  # (batch_size, num_patch, dim_code)
        z_q_res, z_idx_res, z_loss_res, _ = self.codebook3(z_e_res)  # (batch_size, num_patch, dim_code)  Residual-Codebook3

        # Reconstruct time-series
        z_q = z_q_season + z_q_trend + z_q_res  # (batch_size, num_patch, dim_code)
        z_q = self.attr_encoder(z_q)  # (batch_size, num_patch, dim_embedding)
        x_hat, _ = self.decoder(z_q)  # (batch_size, normalize_length)
        x_hat = x_hat * self.x_std + self.x_mean

        # Shape Decoder
        season_hat_norm = self.shape_decoder(z_q_season)  # (bs, num_tokens, patch_size)
        trend_hat_norm = self.shape_decoder(z_q_trend)  # (bs, num_tokens, patch_size)
        bs, num_tokens, patch_size = season_hat_norm.shape
        season_hat_norm = rearrange(season_hat_norm, "B L P -> B (L P)")  # (bs, normalize_length)
        trend_hat_norm = rearrange(trend_hat_norm, "B L P -> B (L P)")  # (bs, normalize_length)
        season_hat = season_hat_norm * self.season_res_std + self.season_res_mean
        trend_hat = trend_hat_norm * self.trend_std + self.trend_mean

        output_dict = {
            'x_true': x,
            'x_pred': x_hat,
            's_season_true': self.season_res,
            's_pred': season_hat,
            's_trend_true': self.trend,
            's_trend_pred': trend_hat,
            'code_trend': z_q_trend,
            'code_idx_trend': z_idx_trend,
            'code_season': z_q_season,
            'code_idx_season': z_idx_season,
            'code_res': z_q_res,
            'code_idx_res': z_idx_res,

        }

        # Compute loss
        if compute_loss:
            # Reconstruction loss
            x_loss = nn.functional.mse_loss(x_hat, x)
            # s = extract_subsequence(x, t_hat, l_hat, self.len_s, smooth=self.s_smooth_factor)
            s_loss_season = nn.functional.mse_loss(season_hat, self.season_res.detach())
            output_dict['s_season_true'] = self.season_res
            s_loss_trend = nn.functional.mse_loss(trend_hat, self.trend.detach())
            output_dict['s_trend_true'] = self.trend
            s_loss = s_loss_season + s_loss_trend
            z_loss = z_loss_season + z_loss_trend + z_loss_res


            loss_dict = {
                'ts_loss': x_loss.unsqueeze(0),
                'vq_loss': z_loss.unsqueeze(0),
                'shape_loss': s_loss.unsqueeze(0),
                'vq_season_loss': z_loss_season.unsqueeze(0),
                'vq_trend_loss': z_loss_trend.unsqueeze(0),
                'vq_res_loss': z_loss_res.unsqueeze(0),
                'shape_season_loss': s_loss_season.unsqueeze(0),
                'shape_trend_loss': s_loss_trend.unsqueeze(0),

            }
            return output_dict, loss_dict
        else:
            return output_dict





