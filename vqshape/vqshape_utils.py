import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


def plot_code_heatmap(code_indices, num_codes, title=''):
    """
    Plots a heatmap for visualizing the use of codes in a vector-quantization model.

    Parameters:
    - code_indices: torch.Tensor, a 2D tensor where each element is a code index.
    - num_codes: int, the total number of different codes.

    The function creates a heatmap where each row represents a different code and
    each column represents a position in the input tensor, showing the frequency of each code.
    """
    code_indices = code_indices.cpu()
    # Initialize a frequency matrix
    codes, counts = torch.unique(code_indices, return_counts=True)
    heatmap = torch.zeros(num_codes).scatter_(-1, codes, counts.float())
    if num_codes <= 64:
        heatmap = heatmap.view(8, -1)
    elif num_codes <= 256:
        heatmap = heatmap.view(16, -1)
    elif num_codes <= 1024:
        heatmap = heatmap.view(32, -1)
    else:
        heatmap = heatmap.view(64, -1)
    # heatmap = heatmap.view(int(np.sqrt(num_codes)), -1)

    heatmap = heatmap.numpy()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency")
    ax.set_title(f'Code Usage Heatmap - step {title}')
    plt.tight_layout()

    return fig


def visualize_shapes(attribute_dict, num_sample=10, num_s_sample=25, title=''):
    '''
    Visualize the decoded shapes and time series.
    attribute_dict: a dict of attributes to visualize. With the following keys:
        x_true: real time series.
        x_pred: reconstructed time series.
        s_true: real subsequences.
        s_pred: decoded shapes.
        t_pred: start times of the shapes.
        l_pred: lengths of the shapes.
        mu_pred: offset of the shapes.
        sigma_pred: standard deviation of the shapes.
    '''
    for k, v in attribute_dict.items():
        attribute_dict[k] = v.float().cpu().numpy()
    
    sample_idx = np.random.randint(0, attribute_dict['x_true'].shape[0], num_sample)

    # Visualize time series and all 64 shapes
    fig = plt.figure(figsize=(30, 4))
    for i, idx in enumerate(sample_idx):
        ax = fig.add_subplot(num_sample//5, 5, i+1)
        ax.plot(np.linspace(0, 1, attribute_dict['x_true'].shape[-1]), attribute_dict['x_true'][idx], color='tab:grey', linewidth=5, alpha=0.3)
        ax.plot(np.linspace(0, 1, attribute_dict['x_true'].shape[-1]), attribute_dict['x_pred'][idx], color='tab:blue', linewidth=5, alpha=0.3)
        for j in range(attribute_dict['t_pred'].shape[1]):
            ts = np.linspace(attribute_dict['t_pred'][idx, j], min(attribute_dict['t_pred'][idx, j]+attribute_dict['l_pred'][idx, j], 1), attribute_dict['s_pred'][idx, j].shape[-1])
            ax.plot(ts, attribute_dict['s_pred'][idx, j])
    plt.tight_layout()

    # Visualize each decoded shape and its corresponding real subsequence
    s_true = rearrange(attribute_dict['s_true'], 'B N L -> (B N) L')
    s_pred = rearrange(attribute_dict['s_pred'], 'B N L -> (B N) L')
    t_pred = rearrange(attribute_dict['t_pred'], 'B N L -> (B N) L')
    l_pred = rearrange(attribute_dict['l_pred'], 'B N L -> (B N) L')
    s_samples_idx = np.random.randint(0, s_true.shape[0], num_s_sample)
    s_fig = plt.figure(figsize=(15, 8))
    for i, idx in enumerate(s_samples_idx):
        ax = s_fig.add_subplot(5, num_s_sample//5, i+1)
        ax.plot(np.linspace(t_pred[idx], t_pred[idx] + l_pred[idx], s_true.shape[-1]), s_true[idx], alpha=0.5)
        ax.plot(np.linspace(t_pred[idx], t_pred[idx] + l_pred[idx], s_pred.shape[-1]), s_pred[idx], alpha=0.5)
    plt.tight_layout()

    return fig, s_fig

def visualize_season_components(output_dict, num_sample=5, title=''):
    '''
    可视化季节性成分和趋势成分的解码结果
    output_dict: 包含以下键的字典
        x_true: 真实时间序列 (batch_size, seq_len)
        x_pred: 重建时间序列 (batch_size, seq_len)
        s_season_true: 真实季节性部分 (batch_size, seq_len)
        s_pred: 解码的季节性部分 (batch_size, seq_len)
        s_trend_true: 真实趋势部分 (batch_size, seq_len)
        s_trend_pred: 解码的趋势部分 (batch_size, seq_len)
        code_idx_season: 季节性码本索引 (batch_size, num_patch)
        code_idx_trend: 趋势码本索引 (batch_size, num_patch)
    '''
     # 转为numpy
    for k, v in output_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.float().cpu().detach().numpy()
    
    batch_size = output_dict['x_true'].shape[0]
    seq_len = output_dict['x_true'].shape[1]
    patch_size = 16
    num_patch = seq_len // patch_size
    if batch_size < num_sample:
        sample_idx = list(range(batch_size))
    else:
        sample_idx = np.random.choice(batch_size, num_sample, replace=False)
    
    # 创建颜色映射 - 为每个patch分配不同颜色
    patch_colors = plt.cm.Set3(np.linspace(0, 1, num_patch))
    
    # 创建时间轴
    time_axis = np.linspace(0, 1, seq_len)
    
    # 创建图形
    fig, axes = plt.subplots(num_sample, 3, figsize=(18, 4*num_sample))
    if num_sample == 1:
        axes = axes.reshape(1, -1)

    # 为每个样本绘制三个图
    for i, idx in enumerate(sample_idx):
        row_axes = axes[i] if num_sample > 1 else axes

        # Trend部分
        ax_trend = row_axes[0]
        true_trend = output_dict['s_trend_true'][idx]
        pred_trend = output_dict['s_trend_pred'][idx]
        # 绘制真实的trend 灰线
        ax_trend.plot(time_axis, true_trend, 'grey', linewidth=2, alpha=0.7, label='True Trend')
        # 绘制预测的trend 蓝线
        ax_trend.plot(time_axis, pred_trend, 'blue', linewidth=2, alpha=0.3, label='Predicted Trend')
        # 用不同颜色绘制每个patch
        for patch in range(num_patch):
            start_idx = patch * patch_size
            end_idx = (patch + 1) * patch_size if patch < num_patch - 1 else seq_len
            patch_time = time_axis[start_idx:end_idx]
            patch_trend = pred_trend[start_idx:end_idx]
            ax_trend.plot(patch_time, patch_trend,
                          color=patch_colors[patch], linewidth=3, alpha=0.7,
                          label=f'Patch {patch+1}' if i == 0 else "")
        ax_trend.set_title(f'Sample {idx} - Trend Component')
        ax_trend.set_xlabel('Time')
        
        ax_trend.set_ylabel('Trend')
        ax_trend.legend(loc='upper right', fontsize=8)
        ax_trend.grid(True, alpha=0.3)
        
        # 2. Seasonal部分
        ax_season = row_axes[1]
        true_season = output_dict['s_season_true'][idx]
        pred_season = output_dict['s_pred'][idx]
        
        # 绘制真实seasonal（灰线）
        ax_season.plot(time_axis, true_season, 'grey', linewidth=2, alpha=0.7, label='True Seasonal')
        
        # 绘制预测seasonal（蓝线）
        ax_season.plot(time_axis, pred_season, 'blue', linewidth=2, alpha=0.3, label='Pred Seasonal')
        
        # 用不同颜色绘制每个patch对应的seasonal部分
        for patch in range(num_patch):
            start_idx = patch * patch_size
            end_idx = (patch + 1) * patch_size if patch < num_patch - 1 else seq_len
            patch_time = time_axis[start_idx:end_idx]
            patch_season = pred_season[start_idx:end_idx]
            
            ax_season.plot(patch_time, patch_season, 
                          color=patch_colors[patch], 
                          linewidth=3, alpha=0.7,
                          label=f'Patch {patch}' if patch == 0 else "")
        
        ax_season.set_title(f'Sample {idx}: Seasonal Component')
        ax_season.set_xlabel('Time')
        ax_season.set_ylabel('Seasonal')
        ax_season.legend(loc='upper right', fontsize=8)
        ax_season.grid(True, alpha=0.3)
        
        # 3. 完整重建序列
        ax_full = row_axes[2]
        true_full = output_dict['x_true'][idx]
        pred_full = output_dict['x_pred'][idx]
        
        # 绘制完整真实序列（灰线）
        ax_full.plot(time_axis, true_full, 'grey', linewidth=2, alpha=0.7, label='True Sequence')
        
        # 绘制完整重建序列（蓝线）
        ax_full.plot(time_axis, pred_full, 'blue', linewidth=2, alpha=0.3, label='Pred Sequence')
        
        # 用不同颜色绘制每个patch对应的重建部分
        # 注意：这里我们展示的是整个重建序列的每个patch部分
        for patch in range(num_patch):
            start_idx = patch * patch_size
            end_idx = (patch + 1) * patch_size if patch < num_patch - 1 else seq_len
            patch_time = time_axis[start_idx:end_idx]
            patch_full = pred_full[start_idx:end_idx]
            
            ax_full.plot(patch_time, patch_full, 
                        color=patch_colors[patch], 
                        linewidth=3, alpha=0.7,
                        label=f'Patch {patch}' if patch == 0 else "")
        
        ax_full.set_title(f'Sample {idx}: Full Reconstruction')
        ax_full.set_xlabel('Time')
        ax_full.set_ylabel('Value')
        ax_full.legend(loc='upper right', fontsize=8)
        ax_full.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # 添加码本索引信息（可选）
    if 'code_idx_season' in output_dict and 'code_idx_trend' in output_dict:
        print("\n码本索引信息:")
        for i, idx in enumerate(sample_idx):
            print(f"\n样本 {idx}:")
            print(f"  季节性码本索引: {output_dict['code_idx_season'][idx].tolist()}")
            print(f"  趋势码本索引: {output_dict['code_idx_trend'][idx].tolist()}")
            if 'code_idx_res' in output_dict:
                print(f"  残差码本索引: {output_dict['code_idx_res'][idx].tolist()}")
    return fig

def visualize_patch_reconstruction(output_dict, sample_idx=0, title=''):
    '''
    详细可视化单个样本的patch级重建
    '''
    # 转换为numpy
    for k, v in output_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.float().cpu().detach().numpy()
    
    seq_len = output_dict['x_true'].shape[1]
    patch_size = 16
    num_patch = seq_len // patch_size
    
    # 创建时间轴
    time_axis = np.linspace(0, 1, seq_len)
    
    # 创建颜色映射
    patch_colors = plt.cm.tab20(np.linspace(0, 1, num_patch))
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 原始序列和各成分
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_axis, output_dict['x_true'][sample_idx], 'k-', linewidth=2, label='Original')
    ax1.plot(time_axis, output_dict['s_trend_true'][sample_idx], 'b--', linewidth=1.5, label='True Trend')
    ax1.plot(time_axis, output_dict['s_season_true'][sample_idx], 'r--', linewidth=1.5, label='True Seasonal')
    ax1.set_title(f'Sample {sample_idx}: Original Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 重建序列和各成分
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time_axis, output_dict['x_pred'][sample_idx], 'k-', linewidth=2, label='Reconstructed')
    ax2.plot(time_axis, output_dict['s_trend_pred'][sample_idx], 'b--', linewidth=1.5, label='Pred Trend')
    ax2.plot(time_axis, output_dict['s_pred'][sample_idx], 'r--', linewidth=1.5, label='Pred Seasonal')
    
    # 标记patch边界
    for patch in range(num_patch):
        x_pos = patch * patch_size / seq_len
        ax2.axvline(x=x_pos, color='grey', linestyle=':', alpha=0.5)
    
    ax2.set_title(f'Sample {sample_idx}: Reconstructed Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Patch级重建误差
    ax3 = plt.subplot(3, 1, 3)
    
    # 计算每个patch的误差
    patch_errors = []
    for patch in range(num_patch):
        start_idx = patch * patch_size
        end_idx = (patch + 1) * patch_size if patch < num_patch - 1 else seq_len
        
        true_patch = output_dict['x_true'][sample_idx, start_idx:end_idx]
        pred_patch = output_dict['x_pred'][sample_idx, start_idx:end_idx]
        
        mse = np.mean((true_patch - pred_patch) ** 2)
        patch_errors.append(mse)
    
    # 绘制误差条
    x_positions = np.arange(num_patch) * patch_size + patch_size/2
    bars = ax3.bar(x_positions / seq_len, patch_errors, 
                  width=patch_size/seq_len * 0.8,
                  color=patch_colors, alpha=0.7)
    
    # 添加误差值标签
    for bar, error in zip(bars, patch_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{error:.4f}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_title(f'Sample {sample_idx}: Patch Reconstruction MSE')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('MSE')
    ax3.set_xticks(x_positions / seq_len)
    ax3.set_xticklabels([f'P{i}' for i in range(num_patch)])
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 打印码本信息
    print(f"\n样本 {sample_idx} 的码本信息:")
    if 'code_idx_season' in output_dict:
        print(f"季节性码本索引: {output_dict['code_idx_season'][sample_idx].tolist()}")
    if 'code_idx_trend' in output_dict:
        print(f"趋势码本索引: {output_dict['code_idx_trend'][sample_idx].tolist()}")
    
    return fig

class Timer:
    def __init__(self):
        self.t = time.time_ns()

    def __call__(self):
        ret = f"Interval: {(time.time_ns() - self.t)/1e6:.1f} ms"
        self.t = time.time_ns()
        return ret


def compute_accuracy(logits, labels):
    """
    Compute the accuracy for multi-class classification.

    Args:
    logits (torch.Tensor): The logits output by the model. Shape: [n_samples, n_classes].
    labels (torch.Tensor): The true labels for the data. Shape: [n_samples].

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    # Get the indices of the maximum logit values along the second dimension (class dimension)
    # These indices correspond to the predicted classes.
    _, predicted_classes = torch.max(logits, dim=1)

    # Compare the predicted classes to the true labels
    correct_predictions = (predicted_classes == labels).float()  # Convert boolean to float

    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()  # Convert to Python scalar


def compute_binary_accuracy(logits, labels):
    """
    Compute the accuracy of binary classification predictions.

    Args:
    logits (torch.Tensor): The logits output by the model. Logits are raw, unnormalized scores.
    labels (torch.Tensor): The true labels for the data.

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    # Convert logits to predictions
    predictions = nn.functional.sigmoid(logits) >= 0.5  # Apply sigmoid and threshold
    labels = labels >= 0.5

    # Compare predictions with true labels
    correct_predictions = (predictions == labels).float()  # Convert boolean to float for summing

    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(labels)

    return accuracy.item()  # Convert to Python scalar


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.05):
    """
    Apply label smoothing to a tensor of binary labels.

    Args:
    labels (torch.Tensor): Tensor of binary labels (0 or 1).
    smoothing (float): Smoothing factor to apply to the labels.

    Returns:
    torch.Tensor: Tensor with smoothed labels.
    """
    # Ensure labels are in float format for the smoothing operation
    labels = labels.float()
    
    # Apply label smoothing
    smoothed_labels = labels * (1 - smoothing) + (1 - labels) * smoothing

    return smoothed_labels


def get_gpu_usage():
    gpu_mem = {}
    for i in range(torch.cuda.device_count()):
        gpu_mem[f'GPU {i}'] = torch.cuda.max_memory_allocated(i)/1e9
        # torch.cuda.reset_peak_memory_stats(i)
    return gpu_mem