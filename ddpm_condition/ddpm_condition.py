import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from model import UNetCond
from ema import EMA


class Diffuser:
    def __init__(
        self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"
    ):
        """DDPMのディフュージョンプロセスを制御するクラス

        Args:
            num_timesteps (int): タイムステップの総数
            beta_start (float): ノイズスケジュールの開始値
            beta_end (float): ノイズスケジュールの終了値
            device (str): 使用するデバイス
        """
        self.num_timesteps = num_timesteps
        self.device = device
        # ノイズスケジュールの設定
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        """画像にノイズを付加する

        Args:
            x_0 (torch.Tensor): 元画像
            t (torch.Tensor): タイムステップ

        Returns:
            tuple: ノイズを付加した画像とノイズ
        """
        assert (t >= 1).all() and (t <= self.num_timesteps).all()

        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels=None):
        """ノイズ除去ステップを実行する

        Args:
            model (UNetCond): UNetCondモデル
            x (torch.Tensor): ノイズの含まれた画像
            t (torch.Tensor): タイムステップ
            labels (torch.Tensor, optional): 条件付けに使用するラベル

        Returns:
            torch.Tensor: ノイズを除去した画像
        """
        assert (t >= 1).all() and (t <= self.num_timesteps).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bars[t_idx].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bars[t_idx - 1].view(-1, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + noise * std

    def sample(self, model, x_shape, labels=None):
        """モデルを使用して画像を生成する

        Args:
            model (UNetCond): UNetCondモデル
            x_shape (tuple): 生成する画像の形状 (batch_size, channels, height, width)
            labels (torch.Tensor, optional): 条件付けに使用するラベル

        Returns:
            torch.Tensor: 生成された画像
        """
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * x_shape[0], device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)

        return x


def main(args):
    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Wandbの初期化
    wandb.init(project="ddpm", name=args.run_name)

    # データの読み込み
    data = np.load(args.data_path)
    labels = np.load(args.label_path)

    # データセットとデータローダーの作成
    dataset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # チャネル数の取得
    num_channels = data.shape[1]

    # ラベルの次元数を設定
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        label_dim = args.num_classes
    else:
        label_dim = (labels.shape[1],)

    # モデルとディフューザーの初期化
    diffuser = Diffuser(args.num_timesteps, device=device)
    model = UNetCond(
        in_ch=num_channels,
        time_embed_dim=args.time_embed_dim,
        img_size=args.img_size,
        label_dim=label_dim,
    )
    model.to(device)

    # EMAモデルの初期化（必要な場合）
    if args.use_ema:
        ema = EMA(beta=args.ema_beta, step_start_ema=args.ema_start_step)
        ema_model = UNetCond(
            in_ch=num_channels,
            time_embed_dim=args.time_embed_dim,
            img_size=args.img_size,
            label_dim=label_dim,
        )
        ema_model.to(device)
        ema.reset_parameters(ema_model, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 学習ループ
    global_step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            x, labels = [b.to(device) for b in batch]
            t = torch.randint(1, args.num_timesteps + 1, (len(x),), device=device)

            # ノイズ付加と予測
            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t, labels)
            loss = torch.nn.functional.mse_loss(noise, noise_pred)

            # 逆伝播
            loss.backward()
            optimizer.step()

            # EMAの更新
            if args.use_ema:
                ema.step_ema(ema_model, model)

            # WandBにLossを記録
            wandb.log({"loss": loss.item()}, step=global_step)
            global_step += 1

        # 指定エポックごとに生成とMSE計算
        if (epoch + 1) % args.save_interval == 0:
            # 評価用のモデルを選択
            eval_model = ema_model if args.use_ema else model
            eval_model.eval()

            with torch.no_grad():
                # サンプル生成
                if isinstance(label_dim, int):
                    # クラスラベルの場合、各クラスのサンプルを生成
                    samples_list = []
                    for label in range(label_dim):
                        sample_labels = torch.tensor(
                            [label] * args.num_samples, device=device
                        )
                        samples = diffuser.sample(
                            eval_model,
                            (
                                args.num_samples,
                                num_channels,
                                args.img_size,
                                args.img_size,
                            ),
                            sample_labels,
                        )
                        samples_list.append(samples)
                    samples = torch.cat(samples_list, dim=0)
                else:
                    # 画像ラベルの場合、指定された条件で生成
                    condition_labels = labels[: args.num_samples].to(device)
                    samples = diffuser.sample(
                        eval_model,
                        (args.num_samples, num_channels, args.img_size, args.img_size),
                        condition_labels,
                    )

                wandb.log(
                    {"generated_samples": wandb.Image(samples), "epoch": epoch},
                    step=global_step,
                )

                # モデルの保存
                if args.save_model:
                    save_dict = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    }
                    if args.use_ema:
                        save_dict["ema_model_state_dict"] = ema_model.state_dict()

                    torch.save(
                        save_dict, f"checkpoints/{args.run_name}_epoch_{epoch}.pt"
                    )

            eval_model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional DDPM Training Script")
    # 既存の引数
    parser.add_argument("--run_name", type=str, required=True, help="実験の名前")
    parser.add_argument("--data_path", type=str, required=True, help="学習データのパス")
    parser.add_argument(
        "--label_path", type=str, required=True, help="ラベルデータのパス"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="バッチサイズ")
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="タイムステップ数"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument(
        "--time_embed_dim", type=int, default=100, help="時間埋め込みの次元"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="生成とMSE計算を行うエポック間隔",
    )
    parser.add_argument("--img_size", type=int, default=64, help="画像サイズ")
    parser.add_argument(
        "--num_samples", type=int, default=20, help="生成するサンプル数"
    )

    # 条件付けに関する引数
    parser.add_argument("--num_classes", type=int, default=None, help="クラス数")

    # EMA関連の引数
    parser.add_argument("--use_ema", action="store_true", help="EMAを使用するかどうか")
    parser.add_argument("--ema_beta", type=float, default=0.995, help="EMAの減衰率")
    parser.add_argument(
        "--ema_start_step", type=int, default=2000, help="EMAの更新を開始するステップ数"
    )

    # その他の引数
    parser.add_argument(
        "--save_model", action="store_true", help="モデルを保存するかどうか"
    )

    args = parser.parse_args()
    main(args)
