import torch
import pickle
import os
import random

from models.TimeMixer import MultiScaleTrendMixing, MultiScaleSeasonMixing

class Args:
    def __init__(self):
        self.seq_len = 96
        self.down_sampling_window = 2
        self.down_sampling_layers = 3

def generate_trend_input(seq_len, down_sampling_layers, batch_size, channels, device):
    """生成多层 trend list，每层是 [B, C, L]，每层 L 减半"""
    trend_list = []
    L = seq_len
    for _ in range(down_sampling_layers + 1):
        t = torch.randn(batch_size, channels, L, device=device)
        trend_list.append(t)
        L = L // 2
    return trend_list

def generate_season_input(seq_len, down_sampling_layers, batch_size, channels, device):
    """生成多层 season list，每层是 [B, C, L]，每层 L 减半"""
    # 生成的结构与trend_list相同，但使用不同的随机种子
    season_list = []
    L = seq_len
    for _ in range(down_sampling_layers + 1):
        t = torch.randn(batch_size, channels, L, device=device)
        season_list.append(t)
        L = L // 2
    return season_list

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Args()

    batch_size = 7
    channels = 16

    # 创建目录
    os.makedirs("tensors", exist_ok=True)

    # ----- MultiScaleTrendMixing 测试数据生成 -----
    trend_list_data = []
    out_trend_list_data = []

    trend_model = MultiScaleTrendMixing(
        seq_len=config.seq_len,
        down_sampling_window=config.down_sampling_window,
        down_sampling_layers=config.down_sampling_layers
    ).to(device)
    trend_model.eval()

    # 保存模型状态
    torch.save(trend_model.state_dict(), "tensors/trend_model_state.pt")

    with torch.no_grad():
        for _ in range(3):
            trend_input = generate_trend_input(
                config.seq_len,
                config.down_sampling_layers,
                batch_size,
                channels,
                device
            )
            out = trend_model(trend_input)

            trend_list_data.append([x.cpu() for x in trend_input])
            out_trend_list_data.append([y.cpu() for y in out])

    with open("tensors/trend_list.pkl", "wb") as f:
        for item in trend_list_data:
            pickle.dump(item, f)

    with open("tensors/out_trend_list.pkl", "wb") as f:
        for item in out_trend_list_data:
            pickle.dump(item, f)

    print("已生成 3 个 trend 测试样例并保存至 tensors/trend_list.pkl 和 tensors/out_trend_list.pkl")
    print("trend 模型状态已保存至 tensors/trend_model_state.pt")

    # ----- MultiScaleSeasonMixing 测试数据生成 -----
    season_list_data = []
    out_season_list_data = []

    season_model = MultiScaleSeasonMixing(
        seq_len=config.seq_len,
        down_sampling_window=config.down_sampling_window,
        down_sampling_layers=config.down_sampling_layers
    ).to(device)
    season_model.eval()

    # 保存模型状态
    torch.save(season_model.state_dict(), "tensors/season_model_state.pt")

    with torch.no_grad():
        for _ in range(3):
            season_input = generate_season_input(
                config.seq_len,
                config.down_sampling_layers,
                batch_size,
                channels,
                device
            )
            out = season_model(season_input)

            season_list_data.append([x.cpu() for x in season_input])
            out_season_list_data.append([y.cpu() for y in out])

    with open("tensors/season_list.pkl", "wb") as f:
        for item in season_list_data:
            pickle.dump(item, f)

    with open("tensors/out_season_list.pkl", "wb") as f:
        for item in out_season_list_data:
            pickle.dump(item, f)

    print("已生成 3 个 season 测试样例并保存至 tensors/season_list.pkl 和 tensors/out_season_list.pkl")
    print("season 模型状态已保存至 tensors/season_model_state.pt")

if __name__ == "__main__":
    main()
