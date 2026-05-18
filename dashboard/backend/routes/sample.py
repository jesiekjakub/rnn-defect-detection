"""Synthetic sample generation endpoint."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from dashboard.backend.schemas import SampleSpec, Sequence
from rnn_defect_detection.data.synthetic import generate_sample

router = APIRouter(prefix="/api/sample", tags=["sample"])


@router.post("", response_model=Sequence)
def generate(spec: SampleSpec) -> Sequence:
    """Generate one synthetic sample matching the spec.

    Per-call seeding uses a fresh ``np.random.Generator``; this avoids polluting
    the global RNG, so calls remain independent regardless of which seed (if
    any) the user picks.
    """
    defects = np.array(spec.defects, dtype=float)
    rng = np.random.default_rng(spec.seed) if spec.seed is not None else None

    sample = generate_sample(spec.seq_len, defects, rng=rng)

    # The signal generator hardcodes a noise scale of 0.2; we apply the
    # user's noise scale as an additional layer rather than re-deriving the
    # synthetic recipe.
    if spec.noise_scale != 0.2:
        extra = (np.random.rand(*sample.shape) if rng is None else rng.random(sample.shape))
        sample = sample + (spec.noise_scale - 0.2) * extra

    return Sequence(
        x=sample.tolist(),
        y_true=[int(v) for v in defects],
        origin="synthetic",
    )
