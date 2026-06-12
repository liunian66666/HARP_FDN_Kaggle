#!/usr/bin/env python3
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - this script runs on Linux server.
    fcntl = None


ROOT = Path("/home/DM24/workspace/Time_Series_Forecasting/HARP_FDN")
LOG_DIR = ROOT / "logs" / "HARP_FDN_optimized_auto_mse_ETT"
SUMMARY = LOG_DIR / "summary.csv"
PRED_LENS = [96, 192, 336, 720]

DATA_NAME = os.environ.get("DATA_NAME", "").strip()
if DATA_NAME not in {"ETTh2", "ETTm1", "ETTm2"}:
    raise SystemExit("DATA_NAME must be one of ETTh2, ETTm1, ETTm2")


@dataclass(frozen=True)
class Config:
    config_id: str
    train_objective: str
    mse_loss_weight: float
    vali_objective: str
    con_cls_1: float
    con_cls_2: float
    con_time: float
    relative_strength_init: float
    posterior_calib_strength_init: float
    correction_strength_init: float
    mixer_d_model: int
    mixer_dropout: float
    support_temperature: float = 1.0
    use_posterior_calib: int = 1


BASE_CONFIGS = [
    Config("s01_mix30_d96_drop05", "mix", 3.0, "mae", 0.03, 0.03, 0.06, -2.5, -3.0, -2.2, 96, 0.05),
    Config("s02_mix25_d96_drop05", "mix", 2.5, "mae", 0.025, 0.025, 0.05, -2.75, -3.5, -2.5, 96, 0.05),
    Config("s03_mix20_d96_drop05", "mix", 2.0, "mae", 0.02, 0.02, 0.04, -3.0, -4.0, -3.0, 96, 0.05),
    Config("s04_mix15_d96_drop05", "mix", 1.5, "mae", 0.015, 0.015, 0.03, -3.5, -5.0, -4.0, 96, 0.05),
    Config("s05_mix10_d96_drop05", "mix", 1.0, "mae", 0.01, 0.01, 0.02, -4.0, -6.0, -5.0, 96, 0.05),
    Config("s06_mae_weak_d96_drop05", "mae", 0.5, "mse", 0.01, 0.01, 0.02, -4.0, -6.0, -5.0, 96, 0.05),
    Config("s07_mix30_d64_drop05", "mix", 3.0, "mae", 0.03, 0.03, 0.06, -2.5, -3.0, -2.2, 64, 0.05),
    Config("s08_mix20_d64_drop05", "mix", 2.0, "mae", 0.02, 0.02, 0.04, -3.0, -4.0, -3.0, 64, 0.05),
    Config("s09_mix15_d64_drop05", "mix", 1.5, "mae", 0.015, 0.015, 0.03, -3.5, -5.0, -4.0, 64, 0.05),
    Config("s10_mix10_d64_drop05", "mix", 1.0, "mae", 0.01, 0.01, 0.02, -4.0, -6.0, -5.0, 64, 0.05),
    Config("s11_mae_weak_d64_drop05", "mae", 0.5, "mse", 0.01, 0.01, 0.02, -4.0, -6.0, -5.0, 64, 0.05),
    Config("s12_mix20_d64_drop10", "mix", 2.0, "mae", 0.02, 0.02, 0.04, -3.0, -4.0, -3.0, 64, 0.10),
]

FIELDNAMES = [
    "timestamp",
    "data",
    "pred_len",
    "stage",
    "config_id",
    "batch",
    "learning_rate",
    "train_objective",
    "mse_loss_weight",
    "vali_objective",
    "con_cls_1",
    "con_cls_2",
    "con_time",
    "relative_strength_init",
    "posterior_calib_strength_init",
    "correction_strength_init",
    "mixer_d_model",
    "mixer_dropout",
    "support_temperature",
    "use_posterior_calib",
    "mse",
    "mae",
    "log_file",
]


def ensure_summary():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not SUMMARY.exists():
        with SUMMARY.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def read_rows():
    ensure_summary()
    with SUMMARY.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(row):
    ensure_summary()
    with SUMMARY.open("a", newline="", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def is_done(pred_len, config_id):
    for row in read_rows():
        if (
            row.get("data") == DATA_NAME
            and row.get("pred_len") == str(pred_len)
            and row.get("config_id") == config_id
            and row.get("mse")
            and row.get("mse") != "NA"
        ):
            return True
    return False


def best_row(pred_len):
    rows = [
        row for row in read_rows()
        if row.get("data") == DATA_NAME and row.get("pred_len") == str(pred_len) and row.get("mse") not in {"", "NA", None}
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["mse"]))


def config_from_row(row):
    return Config(
        config_id=row["config_id"],
        train_objective=row["train_objective"],
        mse_loss_weight=float(row["mse_loss_weight"]),
        vali_objective=row["vali_objective"],
        con_cls_1=float(row["con_cls_1"]),
        con_cls_2=float(row["con_cls_2"]),
        con_time=float(row["con_time"]),
        relative_strength_init=float(row["relative_strength_init"]),
        posterior_calib_strength_init=float(row["posterior_calib_strength_init"]),
        correction_strength_init=float(row["correction_strength_init"]),
        mixer_d_model=int(row["mixer_d_model"]),
        mixer_dropout=float(row["mixer_dropout"]),
        support_temperature=float(row["support_temperature"]),
        use_posterior_calib=int(row["use_posterior_calib"]),
    )


def refine_configs(pred_len):
    row = best_row(pred_len)
    if row is None:
        return []
    base = config_from_row(row)
    other_d = 64 if base.mixer_d_model == 96 else 96
    tighter_con = max(base.con_time * 0.5, 0.01)
    lo_mse = max(base.mse_loss_weight - 0.5, 0.5)
    hi_mse = base.mse_loss_weight + 0.5

    return [
        replace(
            base,
            config_id=f"a01_best_conhalf_{base.mixer_d_model}",
            con_cls_1=max(base.con_cls_1 * 0.5, 0.005),
            con_cls_2=max(base.con_cls_2 * 0.5, 0.005),
            con_time=tighter_con,
            relative_strength_init=base.relative_strength_init - 0.5,
            posterior_calib_strength_init=base.posterior_calib_strength_init - 1.0,
        ),
        replace(base, config_id=f"a02_best_mseup_{base.mixer_d_model}", mse_loss_weight=hi_mse),
        replace(base, config_id=f"a03_best_msedown_{base.mixer_d_model}", mse_loss_weight=lo_mse),
        replace(base, config_id=f"a04_best_temp07_{base.mixer_d_model}", support_temperature=0.7),
        replace(base, config_id=f"a05_best_temp13_{base.mixer_d_model}", support_temperature=1.3),
        replace(base, config_id=f"a06_best_nocalib_{base.mixer_d_model}", use_posterior_calib=0),
        replace(base, config_id=f"a07_best_dswap_{other_d}", mixer_d_model=other_d),
    ]


def run_config(pred_len, config, stage):
    if is_done(pred_len, config.config_id):
        print(f"[{DATA_NAME}] skip done pred_len={pred_len} config={config.config_id}", flush=True)
        return

    model_id = (
        f"HARP_FDN_opt_AUTO_MSE_B1024_{DATA_NAME}_pl{pred_len}_{stage}_{config.config_id}"
    )
    log_file = LOG_DIR / f"{model_id}.log"
    coarse_len = pred_len // 4

    cmd = [
        sys.executable,
        "-u",
        "run_patch0608.py",
        "--is_training", "1",
        "--root_path", "./dataset/ETT-small/",
        "--data_path", f"{DATA_NAME}.csv",
        "--model_id", model_id,
        "--model", "HARP_FDN_optimized",
        "--data", DATA_NAME,
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--enc_in", "7",
        "--des", "AutoMSE",
        "--itr", "1",
        "--train_epochs", "100",
        "--patience", "10",
        "--batch_size", "1024",
        "--learning_rate", "0.0001",
        "--lradj", "sigmoid",
        "--train_objective", config.train_objective,
        "--mse_loss_weight", str(config.mse_loss_weight),
        "--vali_objective", config.vali_objective,
        "--use_vali_ratio", "0",
        "--con_cls_1", str(config.con_cls_1),
        "--con_cls_2", str(config.con_cls_2),
        "--con_time", str(config.con_time),
        "--ma_type", "ema",
        "--alpha", "0.3",
        "--beta", "0.3",
        "--patch_len", "16",
        "--stride", "8",
        "--padding_patch", "end",
        "--support_temperature", str(config.support_temperature),
        "--support_path", "idx2.xlsx",
        "--num_support", "25",
        "--coarse_len", str(coarse_len),
        "--use_relative_decode", "1",
        "--use_distribution_transport", "1",
        "--use_uncertainty_fusion", "1",
        "--use_posterior_calib", str(config.use_posterior_calib),
        "--relative_strength_init", str(config.relative_strength_init),
        "--relative_residual_scale_init", "0.5",
        "--transport_var_weight", "0.05",
        "--posterior_calib_strength_init", str(config.posterior_calib_strength_init),
        "--use_horizon_segment", "1",
        "--use_residual_correction", "1",
        "--use_spectral_transport", "1",
        "--correction_strength_init", str(config.correction_strength_init),
        "--use_dynamic_support", "1",
        "--mixer_d_model", str(config.mixer_d_model),
        "--mixer_layers", "2",
        "--mixer_dropout", str(config.mixer_dropout),
        "--use_checkpoint", "1",
        "--light_head", "1",
        "--resume", "0",
    ]

    print("=" * 80, flush=True)
    print(f"[{DATA_NAME}] stage={stage} pred_len={pred_len} config={config.config_id}", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    print("=" * 80, flush=True)

    metric_re = re.compile(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)")
    last_metric = None
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            match = metric_re.search(line)
            if match:
                last_metric = match.groups()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"run failed with exit code {code}: {model_id}")

    mse, mae = last_metric if last_metric is not None else ("NA", "NA")
    append_row(
        {
            "timestamp": subprocess.check_output(["date", "+%F %T"], text=True).strip(),
            "data": DATA_NAME,
            "pred_len": pred_len,
            "stage": stage,
            "config_id": config.config_id,
            "batch": 1024,
            "learning_rate": 0.0001,
            "train_objective": config.train_objective,
            "mse_loss_weight": config.mse_loss_weight,
            "vali_objective": config.vali_objective,
            "con_cls_1": config.con_cls_1,
            "con_cls_2": config.con_cls_2,
            "con_time": config.con_time,
            "relative_strength_init": config.relative_strength_init,
            "posterior_calib_strength_init": config.posterior_calib_strength_init,
            "correction_strength_init": config.correction_strength_init,
            "mixer_d_model": config.mixer_d_model,
            "mixer_dropout": config.mixer_dropout,
            "support_temperature": config.support_temperature,
            "use_posterior_calib": config.use_posterior_calib,
            "mse": mse,
            "mae": mae,
            "log_file": str(log_file.relative_to(ROOT)),
        }
    )
    print(f"[{DATA_NAME}] recorded pred_len={pred_len} config={config.config_id} mse={mse} mae={mae}", flush=True)


def main():
    ensure_summary()
    for pred_len in PRED_LENS:
        print(f"\n[{DATA_NAME}] ===== base stage pred_len={pred_len} =====", flush=True)
        for config in BASE_CONFIGS:
            run_config(pred_len, config, "base")

        current_best = best_row(pred_len)
        if current_best is not None:
            print(
                f"[{DATA_NAME}] base best pred_len={pred_len}: "
                f"{current_best['config_id']} mse={current_best['mse']} mae={current_best['mae']}",
                flush=True,
            )

        print(f"\n[{DATA_NAME}] ===== refine stage pred_len={pred_len} =====", flush=True)
        for config in refine_configs(pred_len):
            run_config(pred_len, config, "refine")

        current_best = best_row(pred_len)
        if current_best is not None:
            print(
                f"[{DATA_NAME}] final best pred_len={pred_len}: "
                f"{current_best['config_id']} mse={current_best['mse']} mae={current_best['mae']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
