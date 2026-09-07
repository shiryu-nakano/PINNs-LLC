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

`tensorflow[and-cuda]` を指定しているのは、プレーンな `tensorflow` パッケージには
GPU用のCUDA/cuDNN共有ライブラリが同梱されておらず、`nvidia-smi`でGPUが見えていても
TensorFlowからは "Cannot dlopen some GPU libraries" となってGPUが認識されないため。
`[and-cuda]` extraが対応する `nvidia-cudnn-cu12` 等をpipで自動インストールする。

これ以降は `source .venv/bin/activate` してから素の `python ...` で実行する
(次のステップでactivateスクリプトに手を入れるため、`uv run python ...` ではなく
必ずactivateしてから実行すること)。

## 4. GPU認識の確認 + LD_LIBRARY_PATHの恒久対応

`tensorflow[and-cuda]` を入れても、環境によっては `nvidia-*-cu12` のライブラリを
TensorFlowが自動で見つけられず "Cannot dlopen some GPU libraries" のまま
GPUが空リストになることがある(s23で発生・確認済み)。その場合、`LD_LIBRARY_PATH`に
`site-packages/nvidia/*/lib` を明示的に通す必要がある。

シェル全体に`export`すると他プロジェクト(`dsb_llc`等)のCUDA/cuDNNと衝突しうるため、
**このvenvをactivateしている間だけ**有効になるよう `.venv/bin/activate` にフックを追加する
(一度だけでよい。`deactivate`すれば元の`LD_LIBRARY_PATH`に戻る)。

```bash
cat >> .venv/bin/activate <<'EOF'

# --- PINNs-LLC: scope CUDA libs to this venv only ---
_PINNS_LLC_NVIDIA_DIR="$(python -c "import nvidia; print(nvidia.__path__[0])" 2>/dev/null)"
if [ -n "$_PINNS_LLC_NVIDIA_DIR" ]; then
    export _OLD_LD_LIBRARY_PATH_PINNS_LLC="$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$(find "$_PINNS_LLC_NVIDIA_DIR" -maxdepth 2 -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH"
fi

_pinns_llc_old_deactivate="$(declare -f deactivate | tail -n +2)"
deactivate () {
    if [ -n "${_OLD_LD_LIBRARY_PATH_PINNS_LLC+x}" ]; then
        export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH_PINNS_LLC"
        unset _OLD_LD_LIBRARY_PATH_PINNS_LLC
    fi
eval "$_pinns_llc_old_deactivate"
}
EOF
```

追加したら入り直して確認する:

```bash
deactivate
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`[PhysicalDevice(name='/physical_device:GPU:0', ...), ...]` のようにGPUの数だけ
(このサーバーでは8枚)リストが返ればOK。

## 5. データ取得

`run/train.py`は最初に`data_dir/case{N}/data.mat`をGoogle Driveからダウンロードしようとする。
**2026-09時点でCase 1〜4すべてのファイルがGoogle Drive上から削除されており、
ダウンロードは失敗する**(`gdown.exceptions.FileURLRetrievalError`)。

代わりに、元論文([Thakur et al., Physics of Fluids 36, 081915 (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10942483/),
[arXiv:2403.03970](https://arxiv.org/abs/2403.03970))に記載の支配方程式と
拡散係数D(x,y)を使って、パイプライン検証用の代替データを生成できる
(`src/synthetic_data.py`参照。論文の数値ベンチマークの厳密な再現ではなく、
初期条件・境界条件はこちらで仮定した値であることに注意)。

```bash
python run/make_synthetic_data.py --case 1
```

これで `~/server/PINNs-LLC/data/case1/data.mat` が生成される
(`download_data()`と同じパス規則なので、以後`run/train.py --case 1`は
このファイルを検知してダウンロードをスキップする)。

本物の`data.mat`が別途手に入った場合は、同じパスに置き換えれば
そちらが使われる。

## 6. スモークテスト

いきなり本番設定 (`N_iter=320000`) を回さず、まず小規模に配線を確認する。
(`.venv`をactivate済みであること)

```bash
python run/train.py --case 1 --n-iter 100 --log-every 10 --eval-every 50 --ckpt-every 50
```

確認すること:
- `~/server/PINNs-LLC/ckpts/case1/` に `context.npz`, `llc_eval_set.npz`, `ckpt_0000000.npz` などが保存される
- ログにLoss_data / Loss_diff / Loss_fが出力される

問題なければ本番の設定 (`run/configs/example.yaml` など) で実行する。

```bash
python run/train.py --config run/configs/example.yaml
```
