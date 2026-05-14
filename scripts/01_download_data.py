"""scripts/01_download_data.py — orquesta descarga + alineación.

Uso:
    uv run python scripts/01_download_data.py
    uv run python scripts/01_download_data.py --force-redownload

Salidas:
    data/raw/equities/{ticker}.parquet     (6 archivos)
    data/raw/macro/{name}.parquet           (3 archivos)
    data/processed/aligned.parquet          (dataset final)

La fecha de ejecución se loggea para reproducibilidad y debe anotarse en el README.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alignment import PROCESSED_PATH, align_to_xnys  # noqa: E402
from src.data.download import (  # noqa: E402
    EQUITY_TICKERS_YF,
    MACRO_TICKERS_YF,
    RAW_EQUITIES_DIR,
    RAW_MACRO_DIR,
    _safe_filename,
    download_equities,
    download_macro,
)


def _equities_already_downloaded() -> bool:
    return all(
        (RAW_EQUITIES_DIR / f"{_safe_filename(t)}.parquet").exists()
        and (RAW_EQUITIES_DIR / f"{_safe_filename(t)}.parquet").stat().st_size > 0
        for t in EQUITY_TICKERS_YF
    )


def _macros_already_downloaded() -> bool:
    return all(
        (RAW_MACRO_DIR / f"{n}.parquet").exists()
        and (RAW_MACRO_DIR / f"{n}.parquet").stat().st_size > 0
        for n in MACRO_TICKERS_YF
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-descarga aunque los Parquet crudos ya existan",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("F1-T7")

    log.info(
        "Inicio descarga snapshot — fecha %s UTC",
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )

    if not args.force_redownload and _equities_already_downloaded():
        log.info(
            "Equities ya descargados, salto download_equities (--force-redownload para sobrescribir)"
        )
    else:
        download_equities()

    if not args.force_redownload and _macros_already_downloaded():
        log.info("Macros ya descargados, salto download_macro")
    else:
        download_macro()

    log.info("Alineando al calendario XNYS")
    df = align_to_xnys()
    log.info("Fin OK — %s (%d filas × %d cols)", PROCESSED_PATH, *df.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
