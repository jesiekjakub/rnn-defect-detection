"""User CSV / JSON upload endpoint.

Two accepted formats:

* CSV (long): ``sequence_id, t, sensor_0, sensor_1, sensor_2[, defect_0..defect_4]``.
* JSON: ``[{"x": [[s0, s1, s2], ...], "y_true": [0/1, ...]}, ...]``.

Hard limits: 5 MB upload, 200 sequences, 500 timesteps each. Sensor count is
hardcoded to 3 because both models are baked at that input dim.
"""

from __future__ import annotations

import io
import json
from typing import Iterable

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, status

from dashboard.backend.schemas import Sequence, UploadResponse
from rnn_defect_detection.config import (
    MAX_INFERENCE_SEQ_LEN,
    NUM_DEFECT_TYPES,
    NUM_SENSORS,
)

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_BYTES = 5 * 1024 * 1024
MAX_SEQUENCES = 200
SENSOR_COLS = [f"sensor_{i}" for i in range(NUM_SENSORS)]
DEFECT_COLS = [f"defect_{i}" for i in range(NUM_DEFECT_TYPES)]


@router.post("", response_model=UploadResponse)
async def upload_sequences(file: UploadFile) -> UploadResponse:
    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"upload exceeds {MAX_BYTES} bytes",
        )

    filename = (file.filename or "").lower()
    warnings: list[str] = []

    try:
        if filename.endswith(".csv") or (file.content_type and "csv" in file.content_type):
            sequences = _parse_csv(payload, warnings)
        elif filename.endswith(".json") or (file.content_type and "json" in file.content_type):
            sequences = _parse_json(payload, warnings)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="unsupported file type; use .csv or .json",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"parse error: {exc}") from exc

    if not sequences:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no sequences found in upload")
    if len(sequences) > MAX_SEQUENCES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"too many sequences ({len(sequences)} > {MAX_SEQUENCES})",
        )

    return UploadResponse(sequences=sequences, warnings=warnings)


def _parse_csv(payload: bytes, warnings: list[str]) -> list[Sequence]:
    df = pd.read_csv(io.BytesIO(payload))
    missing = [col for col in SENSOR_COLS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"missing sensor columns: {missing}",
        )
    if "sequence_id" not in df.columns:
        df["sequence_id"] = "seq_0"
        warnings.append("no sequence_id column; treated as a single sequence")
    if "t" not in df.columns:
        df = df.assign(t=df.groupby("sequence_id").cumcount())

    has_labels = all(col in df.columns for col in DEFECT_COLS)

    out: list[Sequence] = []
    for seq_id, group in df.groupby("sequence_id", sort=False):
        group = group.sort_values("t")
        if len(group) > MAX_INFERENCE_SEQ_LEN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"sequence '{seq_id}' length {len(group)} exceeds cap {MAX_INFERENCE_SEQ_LEN}",
            )
        x = group[SENSOR_COLS].astype(float).values.tolist()
        y = None
        if has_labels:
            labels = group[DEFECT_COLS].astype(float).iloc[0].astype(int).tolist()
            y = labels
        out.append(Sequence(x=x, y_true=y, sequence_id=str(seq_id), origin="upload"))
    return out


def _parse_json(payload: bytes, warnings: list[str]) -> list[Sequence]:
    obj = json.loads(payload)
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSON root must be a list of {x, y_true?} objects",
        )

    out: list[Sequence] = []
    for i, entry in enumerate(_ensure_dicts(obj)):
        x = entry.get("x")
        if not isinstance(x, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"entry {i}: 'x' must be a list of [s0, s1, s2] rows",
            )
        if len(x) > MAX_INFERENCE_SEQ_LEN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"entry {i}: length {len(x)} exceeds cap {MAX_INFERENCE_SEQ_LEN}",
            )
        y = entry.get("y_true")
        if y is not None and len(y) != NUM_DEFECT_TYPES:
            warnings.append(f"entry {i}: y_true wrong length; dropping ground-truth labels")
            y = None
        out.append(
            Sequence(
                x=[[float(v) for v in row] for row in x],
                y_true=[int(v) for v in y] if y is not None else None,
                sequence_id=str(entry.get("sequence_id", f"seq_{i}")),
                origin="upload",
            )
        )
    return out


def _ensure_dicts(items: Iterable[object]) -> Iterable[dict]:
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"entry {i} is not an object",
            )
        yield item
