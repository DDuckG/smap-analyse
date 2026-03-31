# SMAP Clean Runtime

Clean runtime source for UAP-native social listening with one domain ontology per run.

Bundled domain packs:

- `configs/domains/cosmetics_vn.yaml`
- `configs/domains/beer_vn.yaml`
- `configs/domains/blockchain_vn.yaml`

## Environment

- Python `3.12`
- Windows `cmd.exe`
- Local SQLite support
- Internet access on first Hugging Face model download, or a warm local cache

## Fresh Setup

Run from the repo root:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[intelligence]
mkdir var
mkdir var\data
mkdir var\data\models
mkdir var\data\reports
mkdir var\data\bronze
mkdir var\data\silver
mkdir var\data\gold
mkdir var\data\insights
curl.exe -L "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -o var\data\models\lid.176.ftz
smap db-upgrade
```

## fastText LID Model

Expected path:

```cmd
var\data\models\lid.176.ftz
```

Download command:

```cmd
curl.exe -L "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" -o var\data\models\lid.176.ftz
```

Quick check:

```cmd
dir var\data\models\lid.176.ftz
```

Optional override:

```cmd
set SMAP_INTELLIGENCE__LANGUAGE_ID__FASTTEXT_MODEL_PATH=.\var\data\models\lid.176.ftz
```

## Main Commands

Database upgrade:

```cmd
smap db-upgrade
```

Doctor:

```cmd
smap doctor
```

Validate a batch:

```cmd
smap validate-batch C:\path\to\input.zip
```

Run cosmetics:

```cmd
smap run-pipeline C:\path\to\input.zip --domain-ontology .\configs\domains\cosmetics_vn.yaml
```

Run beer:

```cmd
smap run-pipeline C:\path\to\input.zip --domain-ontology .\configs\domains\beer_vn.yaml
```

Run blockchain:

```cmd
smap run-pipeline C:\path\to\input.zip --domain-ontology .\configs\domains\blockchain_vn.yaml
```

Write pipeline output to JSON:

```cmd
smap run-pipeline C:\path\to\input.zip --domain-ontology .\configs\domains\beer_vn.yaml --output-json .\var\data\reports\pipeline_result.json
```

## Minimal Sanity Checks

```cmd
smap doctor
smap db-upgrade
smap validate-batch C:\path\to\input.zip
```

```cmd
smap run-pipeline C:\path\to\input.zip --domain-ontology .\configs\domains\cosmetics_vn.yaml --output-json .\var\data\reports\sanity_cosmetics.json
```

## Notes

- The runtime is pretrained-only. No training or fine-tuning workflow is included here.
- The runtime writes outputs under `var\`.
- ZIP batch ingestion ignores `__MACOSX` and AppleDouble junk entries.
