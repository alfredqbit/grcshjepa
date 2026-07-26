# GR-CS-HJEPA Chapter 4 Colab Experimental Driver

This repository scaffold supports the Chapter 4 experimental pipeline for the GR-CS-HJEPA dissertation proposal. It is intentionally a starter implementation: smoke-mode runs verify that data generation, H-JEPA pretraining, downstream heads, routing metrics, damage interventions, and statistical utilities execute end to end.

## Main workflow

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_study1_smoke.py --config configs/smoke.yaml
python scripts/run_study2_smoke.py --config configs/smoke.yaml
python scripts/run_study3_smoke.py --config configs/smoke.yaml
```

## Design rule

The notebook is only the driver. The source of truth is `src/`, `tests/`, `configs/`, and `scripts/`.
