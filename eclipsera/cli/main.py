"""Command-line interface for Eclipsera."""

import argparse
import pickle  # nosec B403
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..__version__ import __version__


def main() -> int:
    """Main CLI entry point.

    Returns
    -------
    exit_code : int
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="eclipsera",
        description="Eclipsera - A Modern Machine Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Eclipsera {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show version and system information")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model using AutoML")
    train_parser.add_argument(
        "--data", type=str, required=True, help="Training data path (.npy or .csv)"
    )
    train_parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name (for CSV) or target file (.npy)",
    )
    train_parser.add_argument(
        "--output", type=str, default="model.pkl", help="Output model path (default: model.pkl)"
    )
    train_parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type",
    )
    train_parser.add_argument(
        "--cv", type=int, default=5, help="Cross-validation folds (default: 5)"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument(
        "--model", type=str, required=True, help="Trained model path (.pkl)"
    )
    predict_parser.add_argument(
        "--data", type=str, required=True, help="Input data path (.npy or .csv)"
    )
    predict_parser.add_argument(
        "--confirm-unsafe-pickle",
        action="store_true",
        help="Confirm you trust the pickle model file (required for loading pickle models)",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default="predictions.npy",
        help="Output predictions path (default: predictions.npy)",
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument(
        "--model", type=str, required=True, help="Trained model path (.pkl)"
    )
    evaluate_parser.add_argument(
        "--data", type=str, required=True, help="Test data path (.npy or .csv)"
    )
    evaluate_parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name (for CSV) or target file (.npy)",
    )
    evaluate_parser.add_argument(
        "--confirm-unsafe-pickle",
        action="store_true",
        help="Confirm you trust the pickle model file (required for loading pickle models)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "info":
        return cmd_info()
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    else:
        parser.print_help()
        return 1


def cmd_info() -> int:
    """Show version and system information.

    Returns
    -------
    exit_code : int
        Exit code (0 for success).
    """
    from .. import show_versions

    show_versions()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model using AutoML.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    exit_code : int
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load data
        print(f"Loading data from {args.data}...")
        X, y = _load_data(args.data, args.target)
        print(f"Data shape: {X.shape}, Target shape: {y.shape}")

        # Import AutoML
        if args.task == "classification":
            from ..automl import AutoClassifier

            print("\nTraining classifier with AutoML...")
            model = AutoClassifier(cv=args.cv, verbose=1)
        else:
            from ..automl import AutoRegressor

            print("\nTraining regressor with AutoML...")
            model = AutoRegressor(cv=args.cv, verbose=1)

        # Train
        model.fit(X, y)

        # Save model
        print(f"\nSaving model to {args.output}...")
        with open(args.output, "wb") as f:
            pickle.dump(model, f)

        print(f"\n✓ Training complete!")
        print(f"  Best algorithm: {model.best_algorithm_}")
        print(f"  CV score: {model.best_score_:.4f}")
        print(f"  Model saved to: {args.output}")

        return 0
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Make predictions with a trained model.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    exit_code : int
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load model
        print(f"Loading model from {args.model}...")
        if not getattr(args, "confirm_unsafe_pickle", False):
            print(
                "Error: Loading pickle models requires --confirm-unsafe-pickle flag. "
                "Pickle files can execute arbitrary code. Only load models from trusted sources.",
                file=sys.stderr,
            )
            return 1
        with open(args.model, "rb") as f:
            model = pickle.load(f)  # nosec B301

        # Load data
        print(f"Loading data from {args.data}...")
        X = _load_data(args.data, target=None)
        print(f"Data shape: {X.shape}")

        # Predict
        print("Making predictions...")
        predictions = model.predict(X)

        # Save predictions
        print(f"Saving predictions to {args.output}...")
        np.save(args.output, predictions)

        print(f"\n✓ Predictions complete!")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Saved to: {args.output}")

        return 0
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a trained model.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    exit_code : int
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load model
        print(f"Loading model from {args.model}...")
        if not getattr(args, "confirm_unsafe_pickle", False):
            print(
                "Error: Loading pickle models requires --confirm-unsafe-pickle flag. "
                "Pickle files can execute arbitrary code. Only load models from trusted sources.",
                file=sys.stderr,
            )
            return 1
        with open(args.model, "rb") as f:
            model = pickle.load(f)  # nosec B301

        # Load data
        print(f"Loading data from {args.data}...")
        X, y = _load_data(args.data, args.target)
        print(f"Data shape: {X.shape}, Target shape: {y.shape}")

        # Evaluate
        print("Evaluating model...")
        score = model.score(X, y)
        predictions = model.predict(X)

        print(f"\n✓ Evaluation complete!")
        print(f"  Score: {score:.4f}")

        # Show confusion matrix for classification
        if hasattr(model, "predict_proba"):
            from ..core.metrics import accuracy_score

            accuracy = accuracy_score(y, predictions)
            print(f"  Accuracy: {accuracy:.4f}")

        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        return 1


def _load_data(
    data_path: str, target: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Load data from file.

    Parameters
    ----------
    data_path : str
        Path to data file (.npy or .csv).
    target : str or None
        Target column name (for CSV) or target file path (.npy).
        If None, only returns X.

    Returns
    -------
    data : ndarray or tuple of ndarrays
        If target is None, returns X only.
        Otherwise returns (X, y).
    """
    path = Path(data_path)

    if path.suffix == ".npy":
        X = np.load(data_path, allow_pickle=False)
        if target is None:
            return X
        y = np.load(target, allow_pickle=False)
        return X, y
    elif path.suffix == ".csv":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for CSV support. Install with: pip install pandas"
            )

        df = pd.read_csv(data_path)
        if target is None:
            return df.values

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in CSV. Available columns: {list(df.columns)}"
            )

        y = df[target].values
        X = df.drop(columns=[target]).values
        return X, y
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .npy or .csv")


if __name__ == "__main__":
    sys.exit(main())
