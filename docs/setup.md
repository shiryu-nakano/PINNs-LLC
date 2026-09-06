# セットアップ手順 (s29 / s23)

このリポジトリの学習スクリプト (`run/train.py`) を動かすための環境構築手順。
s29・s23とも同一パス構成なので、この手順はどちらでも同じ。

参照: リポジトリは `~/src/github.com/shiryu-nakano/PINNs-LLC`、データ/チェックポイントは
`~/server/PINNs-LLC` に保存する(README.md参照)。

## 1. 既存環境の確認

```bash
python3 --version
```

`~/server/` 配下には他プロジェクト (`dsb_llc`, `pinns_llc`, `hf_cache` など) もあるため、
それらと衝突しないよう本リポジトリ専用の仮想環境を作る。

## 2. 仮想環境の作成 (uv)

```bash
cd ~/src/github.com/shiryu-nakano/PINNs-LLC
uv venv
```

## 3. 依存パッケージのインストール

```bash
uv pip install -r requirements.txt
```

`requirements.txt` はバージョン未指定 (`tensorflow, numpy, scipy, gdown, pyyaml`)。
サーバーのCUDA/cuDNNバージョンによっては、対応する`tensorflow`のバージョンを
明示的に指定する必要がある場合がある(既存の`dsb_llc`等の環境設定を参考にできる)。

以降のコマンドは `source .venv/bin/activate` してから素の `python ...` で実行するか、
activateせず `uv run python ...` で実行するかのどちらでもよい。

## 4. GPU認識の確認

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

GPUが空リストで返る場合は、CUDA/cuDNNとtensorflowのバージョン不一致を疑う。

## 5. スモークテスト

いきなり本番設定 (`N_iter=320000`) を回さず、まず小規模に配線を確認する。

```bash
uv run python run/train.py --case 1 --n-iter 100 --log-every 10 --eval-every 50 --ckpt-every 50
```

確認すること:
- `~/server/PINNs-LLC/data/case1/data.mat` がダウンロードされる(既にあれば再ダウンロードしない)
- `~/server/PINNs-LLC/ckpts/case1/` に `context.npz`, `llc_eval_set.npz`, `ckpt_0000000.npz` などが保存される
- ログにLoss_data / Loss_diff / Loss_fが出力される

問題なければ本番の設定 (`run/configs/example.yaml` など) で実行する。

```bash
uv run python run/train.py --config run/configs/example.yaml
```
