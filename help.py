import argparse
import torch
import sys

from compressai.entropy_models import EntropyBottleneck
from compressai.models import SymmetricalTransFormer

def parse_cdf():
    parser = argparse.ArgumentParser(description="Helper script.")


def parse_args():
    parser = argparse.ArgumentParser(description="Helper script.")
    sub_parsers = parser.add_subparsers()

    z_cdf_parser = sub_parsers.add_parser("z-cdf", help="Computes an approximation of the cumulative density function of every channel coordinate of z for the given weights")
    z_cdf_parser.add_argument(
        "-m",
        "--model",
        help="Model weights",
    )
    
    args = parser.parse_args()
    return args

def parse_cdf(args):
    parser = argparse.ArgumentParser(description="Computes an approximation of the cumulative density function of every channel coordinate of z for the given weights")

    parser.add_argument(
        "-m",
        "--model",
        help="Model weights",
    )
    
    args = parser.parse_args(args)
    return args

def parse_cmp(args):
    parser = argparse.ArgumentParser(description="Computes an approximation of the cumulative density function of every channel coordinate of z for the given weights")

    parser.add_argument(
        "-m1",
        "--model1",
        help="Model 1 weights",
    )
    parser.add_argument(
        "-m2",
        "--model2",
        help="Model 2 weights",
    )
    
    args = parser.parse_args(args)
    return args

def parse_load(args):
    parser = argparse.ArgumentParser(description="Computes an approximation of the cumulative density function of every channel coordinate of z for the given weights")

    parser.add_argument(
        "-m",
        "--model",
        help="Model weights",
    )
    
    args = parser.parse_args(args)
    return args

def load_bottleneck(path):
    state_dict = torch.load(path)["state_dict"]
    state_dict = {k[7:]: v for k, v in state_dict.items() if k[:7] == "module."}
    filtered = {}

    for k in state_dict:
        if k.startswith("entropy_bottleneck."):
            filtered['.'.join(k.split(".")[1:])] = state_dict[k]

    C = filtered["quantiles"].shape[0]
    bottleneck = EntropyBottleneck(C)
    bottleneck.load_state_dict(filtered)
    return bottleneck

def load_model(path):
    checkpoint = torch.load(path)["state_dict"]
    checkpoint = {k[7:]: v for k, v in checkpoint.items() if k[:7] == "module."}
    model = SymmetricalTransFormer()
    model.load_state_dict(checkpoint, strict=False)
    return model

def main():
    parsers = {
        'cdf': parse_cdf,
        'cmp': parse_cmp,
        'load': parse_load
    }

    cmd = sys.argv[1]
    try:
        args = parsers[cmd](sys.argv[2:])
    except KeyError:
        print(f'Error: unknown command "{cmd}"')

    if cmd == "cdf":
        bottleneck = load_bottleneck(args.model)

        logits = bottleneck._logits_cumulative(bottleneck.quantiles, stop_gradient=True)
        print(bottleneck.quantiles)
        #print(torch.sigmoid(logits))

        #print("Quantiles divergence: ", bottleneck.loss())
    elif cmd == "cmp":
        b1, b2 = load_bottleneck(args.model1), load_bottleneck(args.model2)

        print("Quantiles : ", (b1.quantiles - b2.quantiles).max())
        for i in range(5):
            print(f"Matrix {i}: ", (getattr(b1, f"_matrix{i}") - getattr(b2, f"_matrix{i}")).max())
        for i in range(5):
            print(f"Bias {i}: ", (getattr(b1, f"_bias{i}") - getattr(b2, f"_bias{i}")).max())
        l1 = b1._logits_cumulative(b1.quantiles, stop_gradient=True)
        l2 = b2._logits_cumulative(b2.quantiles, stop_gradient=True)
        print("Logits : ", (l1 - l2).max())
    elif cmd == "load":
        m = SymmetricalTransFormer()

        for name, _ in m.named_parameters():
            print(name)


if __name__ == "__main__":
    main()

    
