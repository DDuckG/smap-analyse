# SMAP v1

Bản này tập trung vào một việc chính:

- nhận đầu vào UAP (`.jsonl`, `.zip`, hoặc thư mục batch)
- chạy full pipeline intelligence
- ghi đầu ra ổn định ra `var/` để team khác có thể wrap thành service

## Yêu cầu môi trường

- Python `3.12`
- Windows / PowerShell hoặc môi trường tương đương
- SQLite hoạt động bình thường
- Có thể tải model từ Hugging Face ở lần chạy đầu tiên, hoặc đã có cache sẵn
- Có model fastText LID `lid.176.ftz`

## Cài đặt mới từ đầu

Chạy tại thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[intelligence]"
python -m alembic upgrade head
Invoke-WebRequest
  -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
  -OutFile ".\var\data\models\lid.176.ftz"
```