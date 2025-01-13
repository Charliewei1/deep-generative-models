# deep generative model

パッケージ管理には[uv](https://docs.astral.sh/uv/)を使用しています。

参考：https://zenn.dev/turing_motors/articles/594fbef42a36ee

`uv add ライブラリ名`でライブラリを追加

`uv remove ライブラリ名`でライブラリを削除

仮想環境を有効化せずとも`uv run main.py`で実行可能

`uv sync`でライブラリのインストールを行う

`.venv/bin/activate`で仮想環境を有効化

`deactivate`で仮想環境を無効化

`uvx ruff check`でFormatter, Linterを実行

---

コードはゼロから作る Deep Learning ❺を参考に作成。

参照：https://github.com/oreilly-japan/deep-learning-from-scratch-5