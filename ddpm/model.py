import math
import torch
from torch import nn


def _pos_encoding(time_idx, output_dim, device="cpu"):
    """単一の時間インデックスに対する位置エンコーディングを計算

    Args:
        time_idx (int): 時間インデックス
        output_dim (int): 出力の次元数
        device (str): 使用するデバイス

    Returns:
        torch.Tensor: 位置エンコーディングベクトル
    """
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


def pos_encoding(timesteps, output_dim, device="cpu"):
    """バッチ内の全時間インデックスに対する位置エンコーディングを計算

    Args:
        timesteps (torch.Tensor): 時間インデックスのバッチ
        output_dim (int): 出力の次元数
        device (str): 使用するデバイス

    Returns:
        torch.Tensor: バッチの位置エンコーディング
    """
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v


class ConvBlock(nn.Module):
    """UNetの畳み込みブロック

    時間埋め込みを考慮した畳み込み層とバッチ正規化を含む
    """

    def __init__(self, in_ch, out_ch, time_embed_dim):
        """
        Args:
            in_ch (int): 入力チャネル数
            out_ch (int): 出力チャネル数
            time_embed_dim (int): 時間埋め込みの次元数
        """
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch), nn.ReLU(), nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        """
        Args:
            x (torch.Tensor): 入力特徴マップ
            v (torch.Tensor): 時間埋め込み

        Returns:
            torch.Tensor: 出力特徴マップ
        """
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class UNet(nn.Module):
    """時間条件付きのUNetモデル

    エンコーダ-デコーダ構造を持ち、各層で時間情報を考慮
    """

    def __init__(self, in_ch=1, time_embed_dim=100, img_size=64, base_channels=64):
        """
        Args:
            in_ch (int): 入力チャネル数
            time_embed_dim (int): 時間埋め込みの次元数
            img_size (int): 入力画像のサイズ（デフォルト: 64）
            base_channels (int): 基本チャネル数（デフォルト: 64）
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # エンコーダ部分
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)

        # ボトルネック部分
        self.bot1 = ConvBlock(128, 256, time_embed_dim)

        # デコーダ部分
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)

        # 出力層
        self.out = nn.Conv2d(64, in_ch, 1)

        # ダウンサンプリングとアップサンプリング
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, timesteps):
        """
        Args:
            x (torch.Tensor): 入力画像
            timesteps (torch.Tensor): タイムステップ

        Returns:
            torch.Tensor: 予測されたノイズ
        """
        # 時間埋め込みの計算
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        # エンコーダパス
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        # ボトルネック
        x = self.bot1(x, v)

        # デコーダパス（スキップ接続を含む）
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)

        # 出力層
        x = self.out(x)
        return x
