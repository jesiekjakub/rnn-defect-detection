"""Command-line entry point: train/eval/predict for both approaches.

Usage:
    python -m rnn_defect_detection train --approach attention --out models/attention.pt
    python -m rnn_defect_detection train --approach seq2seq --quick
    python -m rnn_defect_detection eval  --approach attention --checkpoint models/attention.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from rnn_defect_detection.training import train_attention, train_seq2seq

DEFAULT_MODELS_DIR = Path("models")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rnn-defect-detection",
        description="Train and evaluate the defect-detection models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train one of the approaches")
    train_p.add_argument("--approach", choices=["attention", "seq2seq"], required=True)
    train_p.add_argument("--n-samples", type=int, default=50_000)
    train_p.add_argument("--epochs", type=int, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default=None, help="cpu / cuda / cuda:0; auto by default")
    train_p.add_argument("--out", type=Path, default=None)
    train_p.add_argument("--metrics-out", type=Path, default=DEFAULT_MODELS_DIR / "metrics.json")
    train_p.add_argument("--quick", action="store_true", help="10k samples × 3 epochs for demos")

    eval_p = sub.add_parser("eval", help="Evaluate a saved checkpoint")
    eval_p.add_argument("--approach", choices=["attention", "seq2seq"], required=True)
    eval_p.add_argument("--checkpoint", type=Path, required=True)
    eval_p.add_argument("--n-samples", type=int, default=10_000)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--device", default=None)

    args = parser.parse_args(argv)

    if args.command == "train":
        return _train(args)
    if args.command == "eval":
        return _eval(args)
    parser.print_help()
    return 1


def _train(args: argparse.Namespace) -> int:
    if args.quick:
        n_samples = 10_000
        epochs = 3
    else:
        n_samples = args.n_samples
        epochs = args.epochs

    out_path: Path = args.out or (
        DEFAULT_MODELS_DIR / ("attention_lstm.pt" if args.approach == "attention" else "seq2seq.pt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    if args.approach == "attention":
        result = train_attention(
            n_samples=n_samples,
            epochs=epochs or 5,
            batch_size=args.batch_size or 32,
            seed=args.seed,
            device=args.device,
        )
        torch.save(
            {
                "approach": "attention",
                "model_state": result.model.state_dict(),
                "config": {"hidden_size": result.model.lstm.hidden_size},
            },
            out_path,
        )
        metrics_payload = {
            "approach": "attention",
            "train_losses": result.train_losses,
            "eval_metrics": result.eval_metrics,
            "per_class_metrics": result.per_class_metrics,
        }
    else:
        result = train_seq2seq(
            n_samples=n_samples,
            ae_epochs=epochs or 10,
            clf_epochs=epochs or 10,
            batch_size=args.batch_size or 64,
            seed=args.seed,
            device=args.device,
        )
        torch.save(
            {
                "approach": "seq2seq",
                "autoencoder_state": result.autoencoder.state_dict(),
                "classifier_state": result.classifier.state_dict(),
                "config": {
                    "ae_hidden": result.autoencoder.encoder.hidden_size,
                    "clf_hidden": result.classifier.lstm.hidden_size,
                },
            },
            out_path,
        )
        metrics_payload = {
            "approach": "seq2seq",
            "ae_losses": result.ae_losses,
            "clf_losses": result.clf_losses,
            "eval_metrics": result.eval_metrics,
            "per_class_metrics": result.per_class_metrics,
        }

    # Merge metrics across approaches into a single metrics.json so the
    # dashboard can read both training histories from one file.
    existing: dict[str, object] = {}
    if args.metrics_out.exists():
        try:
            existing = json.loads(args.metrics_out.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing[args.approach] = metrics_payload
    args.metrics_out.write_text(json.dumps(existing, indent=2))

    print(f"\nSaved checkpoint to {out_path}")
    print(f"Saved metrics to {args.metrics_out}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    from rnn_defect_detection.config import HIDDEN_SIZE, NUM_DEFECT_TYPES, NUM_SENSORS
    from rnn_defect_detection.data import SequenceDataset, generate_dataset
    from rnn_defect_detection.models import AttentionLSTM, DefectClassifier, RecurrentAutoencoder
    from rnn_defect_detection.training.attention import _collate as _attn_collate

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.approach == "attention":
        model = AttentionLSTM(hidden_size=checkpoint["config"]["hidden_size"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        x_list, y_list = generate_dataset(args.n_samples, seed=args.seed)
        from torch.utils.data import DataLoader

        from rnn_defect_detection.training.attention import _evaluate

        loader = DataLoader(SequenceDataset(x_list, y_list), batch_size=32, collate_fn=_attn_collate)
        result = _evaluate(model, loader, device, train_losses=[])
        print(json.dumps(result.eval_metrics, indent=2))
        print(json.dumps(result.per_class_metrics, indent=2))
        return 0

    # Approach 2 eval — full pipeline reproduces test metrics from the trained checkpoint.
    cfg = checkpoint["config"]
    ae = RecurrentAutoencoder(input_dim=NUM_SENSORS, hidden_dim=cfg["ae_hidden"]).to(device)
    clf = DefectClassifier(
        input_dim=NUM_SENSORS * 3,
        hidden_dim=cfg["clf_hidden"],
        num_classes=NUM_DEFECT_TYPES,
    ).to(device)
    ae.load_state_dict(checkpoint["autoencoder_state"])
    clf.load_state_dict(checkpoint["classifier_state"])
    ae.eval()
    clf.eval()

    from rnn_defect_detection.inference import predict_seq2seq

    x_list, y_list = generate_dataset(args.n_samples, seed=args.seed)
    correct = 0
    for x, y in zip(x_list, y_list):
        explanation = predict_seq2seq(ae, clf, x, device=device)
        binary = [1 if p > 0.5 else 0 for p in explanation.probs]
        if binary == list(map(int, y)):
            correct += 1
    print(json.dumps({"exact_match_accuracy": correct / len(x_list)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
