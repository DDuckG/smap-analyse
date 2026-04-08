## Setup

From a fresh clone in `cmd.exe`:

```cmd
python -m venv .venv
.venv\Scripts\activate
make install
```

Download the fastText language ID model to `var\data\models\lid.176.ftz`:

```cmd
curl.exe -L "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -o var\data\models\lid.176.ftz
```

The PhoBERT ONNX export is cached under `var\data\intelligence\onnx_models\vinai__phobert-base-v2\` when the runtime is prepared.

## Run

Upgrade the metadata database:

```cmd
make db-upgrade
```

Check runtime readiness:

```cmd
make doctor
```

Run a real input batch with one of the supported domain packs:

```cmd
make run-cosmetics INPUT=path\to\input.jsonl
make run-beer INPUT=path\to\input.jsonl
make run-blockchain INPUT=path\to\input.jsonl
```

Clear runtime outputs under `var\` without touching source files:

```cmd
make clean-runtime
```

## Outputs

Each run writes the normal runtime artifacts under `var\`, including:

- `var\data\reports\run_manifest.json`
- `var\data\reports\metrics.json`
- `var\data\reports\bi_reports.json`
- parquet outputs under `var\data\silver\`
- parquet outputs under `var\data\gold\`
- marts outputs under `var\data\gold\marts\`
