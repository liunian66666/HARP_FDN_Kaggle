# HARP_FDN final package

This directory contains the regenerated final HARP_FDN_PCX model and training scripts used in the current experiments.

## Core files

- `models/HARP_FDN_PCX.py`: final HARP-FDN-PCX model.
- `run.py`: training entry with HARP/PCX arguments.
- `exp/exp_main.py`: model registry and training objective support.
- `idx2.xlsx`: interleaved support points used by distribution decoding.

## Scripts

- `scripts/HARP_FDN/Weather_PCX_samecfg.sh`: Weather 96/192/336/720.
- `scripts/HARP_FDN/ETT_PCX_samecfg.sh`: ETTh1/ETTh2/ETTm1/ETTm2 96/192/336/720.
- `scripts/HARP_FDN/4090D_Solar_ECL_Traffic_PCX.sh`: Solar/ECL/Traffic with larger batch and auto retry.

## Run examples

```bash
cd /home/DM24/workspace/Time_Series_Forecasting/HARP_FDN
bash scripts/HARP_FDN/Weather_PCX_samecfg.sh
bash scripts/HARP_FDN/ETT_PCX_samecfg.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/HARP_FDN/4090D_Solar_ECL_Traffic_PCX.sh
```

## Logs

All new logs are written to:

```bash
logs/HARP_FDN_PCX/
```

Summarize results:

```bash
grep -H "^mse:" logs/HARP_FDN_PCX/*.log
```

## Notes

The 4090D script no longer waits for a specific GPU name. It defaults to `CUDA_VISIBLE_DEVICES=0`, matching the queue backend where the assigned GPU is exposed as GPU 0.
