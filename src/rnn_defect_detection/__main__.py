"""Allow ``python -m rnn_defect_detection ...`` to dispatch to the CLI."""

from rnn_defect_detection.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
