class EMA:
    """Exponential Moving Average (EMA) for model parameters

    モデルパラメータの指数移動平均を計算・管理するクラス。
    学習の安定性向上のために使用される。
    """

    def __init__(self, beta=0.995, step_start_ema=2000):
        """
        Args:
            beta (float): EMAの減衰率（0から1の値）
            step_start_ema (int): EMAの更新を開始するステップ数
        """
        self.beta = beta
        self.step = 0
        self.step_start_ema = step_start_ema

    def update_model_average(self, ma_model, current_model):
        """EMAモデルのパラメータを更新

        Args:
            ma_model (torch.nn.Module): EMAを適用するモデル
            current_model (torch.nn.Module): 現在の学習中のモデル
        """
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """個々のパラメータの移動平均を更新

        Args:
            old (torch.Tensor): 古いパラメータ値
            new (torch.Tensor): 新しいパラメータ値

        Returns:
            torch.Tensor: 更新された移動平均値
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        """EMAの更新ステップを実行

        Args:
            ema_model (torch.nn.Module): EMAを適用するモデル
            model (torch.nn.Module): 現在の学習中のモデル
        """
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """EMAモデルのパラメータをリセット

        Args:
            ema_model (torch.nn.Module): リセットするEMAモデル
            model (torch.nn.Module): パラメータのコピー元となる現在のモデル
        """
        ema_model.load_state_dict(model.state_dict())
