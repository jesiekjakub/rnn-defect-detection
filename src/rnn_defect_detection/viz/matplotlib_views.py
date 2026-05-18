"""Matplotlib plots for both approaches; preserved from the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from rnn_defect_detection.config import DEFECT_COLORS, DEFECT_NAMES, NUM_DEFECT_TYPES
from rnn_defect_detection.inference.attention import analyze_root_cause
from rnn_defect_detection.inference.seq2seq import predict_seq2seq
from rnn_defect_detection.models import AttentionLSTM, DefectClassifier, RecurrentAutoencoder


def visualize_attention_sample(
    model: AttentionLSTM,
    sequence: np.ndarray,
    true_labels: np.ndarray,
    sample_idx: int,
    device: torch.device | str = "cpu",
) -> plt.Figure:
    """Reproduce the notebook's three-part Approach 1 visualization.

    Bar chart of probabilities, sensor signals with attention overlay, attention
    heatmap, and a root-cause text summary.
    """
    model.eval()
    x = torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([sequence.shape[0]], dtype=torch.long)
    with torch.no_grad():
        preds, attentions = model(x, lengths=lengths, return_attention=True)

    probs = preds[0].cpu().numpy()
    attn = attentions[0].cpu().numpy()
    detected = [i for i, p in enumerate(probs) if p > 0.5]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1])

    ax_bar = fig.add_subplot(gs[0, 0])
    positions = np.arange(NUM_DEFECT_TYPES)
    width = 0.35
    ax_bar.bar(positions - width / 2, true_labels, width, label="True", color="lightgray")
    ax_bar.bar(positions + width / 2, (probs > 0.5).astype(int), width, label="Pred", color="steelblue", alpha=0.8)
    for i, p in enumerate(probs):
        ax_bar.text(i + width / 2, (probs > 0.5)[i] + 0.05, f"{p:.2f}", ha="center", fontsize=8)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([f"Type {i}" for i in range(NUM_DEFECT_TYPES)])
    ax_bar.set_title(f"Sample {sample_idx}: predictions", fontsize=10, fontweight="bold")
    ax_bar.legend(loc="upper right", fontsize="small")
    ax_bar.grid(axis="y", alpha=0.3)

    ax_sig = fig.add_subplot(gs[1, :])
    sensor_colors = ["tab:blue", "tab:green", "tab:orange"]
    for s in range(3):
        ax_sig.plot(sequence[:, s], label=f"Sensor {s}", color=sensor_colors[s], alpha=0.7)
    for d_idx in detected:
        ax_sig.fill_between(
            range(sequence.shape[0]),
            sequence.min() - 1,
            sequence.max() + 1,
            where=(attn[:, d_idx] > 0.05),
            color=DEFECT_COLORS[d_idx],
            alpha=0.25,
            label=f"Attn: {DEFECT_NAMES[d_idx]}",
        )
    ax_sig.set_title("Sensor signals + attention activity", fontsize=10, fontweight="bold")
    ax_sig.set_ylabel("Signal value")
    ax_sig.legend(loc="upper right", fontsize="small", ncol=2)
    ax_sig.grid(True, alpha=0.3)

    ax_heat = fig.add_subplot(gs[0, 1])
    if detected:
        im = ax_heat.imshow(attn[:, detected].T, aspect="auto", cmap="Reds", interpolation="nearest")
        ax_heat.set_yticks(range(len(detected)))
        ax_heat.set_yticklabels([DEFECT_NAMES[i] for i in detected])
        ax_heat.set_title("Attention heatmap (detected)", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax_heat)
    else:
        ax_heat.text(0.5, 0.5, "No defects detected", ha="center")
        ax_heat.axis("off")

    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")
    lines = [f"Root cause analysis (sample {sample_idx})", "-" * 60]
    if not detected:
        lines.append("No defects detected.")
    else:
        for d_idx in detected:
            analysis = analyze_root_cause(model, sequence, d_idx, device=device)
            if analysis is None:
                continue
            top_sensor = int(np.argmax(analysis.sensor_importance))
            lines.append(
                f"  {analysis.defect_name} (conf {analysis.confidence:.1%}): "
                f"timesteps {analysis.ranges}; main sensor {top_sensor}."
            )
    ax_text.text(
        0.01, 0.9, "\n".join(lines),
        va="top", fontfamily="monospace", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.2),
    )

    plt.tight_layout()
    return fig


def visualize_seq2seq_sample(
    autoencoder: RecurrentAutoencoder,
    classifier: DefectClassifier,
    sequence: np.ndarray,
    sample_idx: int,
    device: torch.device | str = "cpu",
) -> plt.Figure | None:
    """Plot Approach 2's accepted regions overlaid on the signal, plus derivatives.

    Returns None when the classifier finds nothing globally; this matches the
    notebook's habit of skipping uninteresting samples.
    """
    explanation = predict_seq2seq(autoencoder, classifier, sequence, device=device)
    if not explanation.accepted_regions:
        return None

    residual = np.asarray(explanation.residual)
    velocity = np.asarray(explanation.velocity)
    global_decisions = (np.asarray(explanation.probs) > 0.5).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(sequence, alpha=0.5)
    axes[0].set_title(f"Sample {sample_idx}: best region per defect | global: {global_decisions}")
    axes[0].legend(["S0", "S1", "S2"], loc="upper right", fontsize="x-small")
    for region in explanation.accepted_regions:
        color = DEFECT_COLORS[region.defect_index]
        axes[0].axvspan(region.start, region.end, color=color, alpha=0.15)
        local_residual = residual[region.start : region.end + 1]
        leading_sensor = int(np.argmax(np.max(local_residual, axis=0)))
        axes[0].plot(
            np.arange(region.start, region.end + 1),
            sequence[region.start : region.end + 1, leading_sensor],
            color=color,
            lw=2,
        )
        axes[0].text(
            region.start,
            sequence.max() + 0.1 + region.defect_index * 0.15,
            region.defect_name,
            color=color,
            fontsize=8,
            fontweight="bold",
        )

    axes[1].plot(residual, alpha=0.7)
    axes[1].set_title("Residuals")
    axes[1].grid(alpha=0.3)

    axes[2].plot(velocity, alpha=0.7)
    axes[2].set_title("Velocity")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
