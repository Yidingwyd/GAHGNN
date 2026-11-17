
import argparse
import sys
sys.dont_write_bytecode = True
from training_components import VARIANTS
from train_joint import train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--variables", default=None, help="Optional comma-separated variable order")
    parser.add_argument("--exclude-effect", "--exclude_effect", "--exclude-effects",
                        dest="excluded_effects", nargs="+", action="append", default=[],
                        metavar="VARIABLE",
                        help="Force one exact native effect to zero, e.g. --exclude-effect Al Cr. "
                             "Repeat for multiple effects; omitted means no prior exclusions. "
                             "Higher-order effects containing these variables are not excluded automatically.")
    parser.add_argument("--savepath", required=True)
    parser.add_argument("--size", required=True, type=int, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1100)
    parser.add_argument("--polish-steps", "--polish_steps", type=int, default=40)
    parser.add_argument("--ablation", choices=VARIANTS, default="full")
    parser.add_argument("-b", "--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--disable-cuda", action="store_true")
    args = parser.parse_args()
    if args.epochs < 2 or args.polish_steps < 0 or args.batch_size < 1:
        parser.error("epochs >= 2, polish-steps >= 0, and batch-size >= 1 are required.")
    return args


if __name__ == "__main__":
    train(parse_args())
