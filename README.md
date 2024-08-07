# deep generative model

自分用のコードの整理
```
poetry source add torch_cu117 --priority=explicit https://download.pytorch.org/whl/cu117
poetry add torch=2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --source torch_cu117
poetry install
poetry shell