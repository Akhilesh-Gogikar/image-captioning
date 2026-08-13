# X-ray image captioning

Django demo for comparing a submitted chest X-ray image against stored image embeddings and returning the caption for the nearest training image. It is useful for visitors evaluating a simple retrieval baseline for biomedical image-captioning experiments, not as a diagnostic system.

## What is included

- `xray_caption/views.py`: `detect` endpoint that accepts an uploaded image or image URL, embeds it with `img2vec`, compares it with `raw_embeddings.npy`, and returns JSON with the nearest caption.
- `bmc_api/urls.py`: routes `POST /xray_caption/detect/` to the caption endpoint.
- `test_images/`: sample CXR images plus the TSV used by the test harness.
- `conda_env.txt` and `requirements.txt`: historical Python environment references.

## Local orientation

The code expects trained assets at the original absolute path used by the author:

`/home/akhilesh/bmc_api/image-captioning/xray_caption/`

If you run it elsewhere, update those paths in `xray_caption/views.py` or mirror that directory layout before starting Django with `manage.py`.

## Example request shape

The endpoint accepts either a multipart form field named `image` or a form field named `url`; successful responses include:

```json
{"success": true, "caption": "..."}
```

## Bundled sample check

Before wiring the historical Django environment, visitors can confirm the
included sample set is present from the repository root:

```bash
python3 - <<'PY'
from pathlib import Path
samples = sorted(Path("test_images").glob("CXR*.png"))
rows = Path("test_images/iu_xray.tsv").read_text().splitlines()
assert len(samples) == 5, len(samples)
assert len(rows) >= len(samples), len(rows)
assert all(p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for p in samples)
print(f"{len(samples)} sample images and {len(rows)} TSV rows found")
PY
```

## Repository note

This repository previously had both `README.md` and `Readme.MD`, which collide on case-insensitive filesystems. The project documentation now lives in `README.md` only.

## Citation

If you reference this repository in a project, paper, or demo write-up, cite the GitHub repository:

```bibtex
@software{gogikar_xray_image_captioning,
  author = {Gogikar, Akhilesh},
  title = {X-ray image captioning},
  url = {https://github.com/Akhilesh-Gogikar/image-captioning}
}
```
