"""CLI entry point for the AMC experiment."""

import argparse

from evaluation import cmd_eval
from plotting import cmd_viz
from training import cmd_train


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")

    ap_train = sub.add_parser("train")
    ap_train.add_argument(
        "--runs",
        nargs="+",
        default=["complex_moment", "complex_narrow", "real_narrow", "real_full"],
    )
    ap_train.add_argument("--epochs", type=int, default=None)
    ap_train.add_argument("--out_dir", type=str, default=None)
    ap_train.set_defaults(func=cmd_train)

    ap_eval = sub.add_parser("eval")
    ap_eval.add_argument("--model_dir", default="trained_modclass")
    ap_eval.add_argument("--results_dir", default="results")
    ap_eval.add_argument("--viz_dir", default="visualizations")
    ap_eval.add_argument("--n", type=int, default=2000)
    ap_eval.set_defaults(func=cmd_eval)

    ap_viz = sub.add_parser("viz")
    ap_viz.add_argument("--viz_dir", default="visualizations")
    ap_viz.set_defaults(func=cmd_viz)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
