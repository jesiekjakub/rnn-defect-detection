"""Static architecture descriptors for the Architecture section."""

from __future__ import annotations

from fastapi import APIRouter

from dashboard.backend.schemas import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureLayer,
    ArchitectureNode,
    ArchitectureResponse,
)
from rnn_defect_detection.config import HIDDEN_SIZE, NUM_DEFECT_TYPES, NUM_SENSORS

router = APIRouter(prefix="/api/architecture", tags=["architecture"])


def _attention_graph() -> ArchitectureGraph:
    return ArchitectureGraph(
        name="Approach 1 — Bi-LSTM + per-class attention",
        nodes=[
            ArchitectureNode(
                id="input",
                label="Input",
                layers=[ArchitectureLayer(name="x", kind="tensor", params={"shape": [None, NUM_SENSORS]})],
                notes=f"{NUM_SENSORS} sensor channels, variable length.",
            ),
            ArchitectureNode(
                id="bilstm",
                label="Bi-directional LSTM",
                layers=[
                    ArchitectureLayer(
                        name="lstm",
                        kind="LSTM",
                        params={"hidden": HIDDEN_SIZE, "layers": 2, "bidirectional": True},
                    )
                ],
                notes=f"Output dim 2 × {HIDDEN_SIZE} per timestep.",
            ),
            ArchitectureNode(
                id="attention",
                label=f"{NUM_DEFECT_TYPES} independent attention heads",
                layers=[
                    ArchitectureLayer(
                        name=f"attn_{i}",
                        kind="MLP",
                        params={"hidden": HIDDEN_SIZE, "output": 1, "softmax_over": "time"},
                    )
                    for i in range(NUM_DEFECT_TYPES)
                ],
                notes="One attention vector per defect class.",
            ),
            ArchitectureNode(
                id="classifiers",
                label=f"{NUM_DEFECT_TYPES} binary classifiers",
                layers=[
                    ArchitectureLayer(
                        name=f"clf_{i}",
                        kind="MLP",
                        params={"hidden": 64, "output": 1, "activation": "sigmoid"},
                    )
                    for i in range(NUM_DEFECT_TYPES)
                ],
            ),
            ArchitectureNode(
                id="output",
                label="Output",
                layers=[ArchitectureLayer(name="probs", kind="tensor", params={"shape": [NUM_DEFECT_TYPES]})],
            ),
        ],
        edges=[
            ArchitectureEdge(src="input", dst="bilstm"),
            ArchitectureEdge(src="bilstm", dst="attention", label="encoder_out"),
            ArchitectureEdge(src="attention", dst="classifiers", label="context"),
            ArchitectureEdge(src="classifiers", dst="output"),
        ],
    )


def _seq2seq_graph() -> ArchitectureGraph:
    return ArchitectureGraph(
        name="Approach 2 — Seq2Seq autoencoder + feature classifier",
        nodes=[
            ArchitectureNode(
                id="input",
                label="Input",
                layers=[ArchitectureLayer(name="x", kind="tensor", params={"shape": [None, NUM_SENSORS]})],
            ),
            ArchitectureNode(
                id="ae",
                label="Autoencoder (healthy-only)",
                layers=[
                    ArchitectureLayer(name="encoder", kind="LSTM", params={"hidden": 64}),
                    ArchitectureLayer(name="decoder", kind="LSTM", params={"hidden": 64}),
                    ArchitectureLayer(name="proj", kind="Linear", params={"out": NUM_SENSORS}),
                ],
                notes="Trained to reconstruct healthy signals; residual flags anomalies.",
            ),
            ArchitectureNode(
                id="features",
                label="Engineered features",
                layers=[
                    ArchitectureLayer(name="original", kind="tensor", params={"channels": NUM_SENSORS}),
                    ArchitectureLayer(name="residual", kind="tensor", params={"channels": NUM_SENSORS}),
                    ArchitectureLayer(name="velocity", kind="tensor", params={"channels": NUM_SENSORS}),
                ],
                notes="Stacked → 9 channels per timestep.",
            ),
            ArchitectureNode(
                id="clf",
                label="LSTM defect classifier",
                layers=[
                    ArchitectureLayer(name="lstm", kind="LSTM", params={"hidden": 64}),
                    ArchitectureLayer(
                        name="fc",
                        kind="Linear",
                        params={"in": 64, "out": NUM_DEFECT_TYPES, "activation": "sigmoid"},
                    ),
                ],
            ),
            ArchitectureNode(
                id="regions",
                label="Region proposal + verification",
                layers=[
                    ArchitectureLayer(name="propose", kind="heuristic", params={"signals": ["residual", "velocity"]}),
                    ArchitectureLayer(name="verify", kind="local_inference", params={"agg": "consensus"}),
                ],
                notes="Local crops are re-classified; only consensus regions are kept.",
            ),
        ],
        edges=[
            ArchitectureEdge(src="input", dst="ae"),
            ArchitectureEdge(src="input", dst="features", label="orig"),
            ArchitectureEdge(src="ae", dst="features", label="residual"),
            ArchitectureEdge(src="input", dst="features", label="velocity"),
            ArchitectureEdge(src="features", dst="clf"),
            ArchitectureEdge(src="clf", dst="regions", label="global preds"),
            ArchitectureEdge(src="features", dst="regions", label="local crops"),
        ],
    )


@router.get("", response_model=ArchitectureResponse)
def get_architectures() -> ArchitectureResponse:
    return ArchitectureResponse(approaches=[_attention_graph(), _seq2seq_graph()])
