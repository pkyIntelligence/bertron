import torch

import sys
sys.path.append('tacotron2/waveglow/')

import argparse

"""
Converts the published waveglow model to a state_dict for easier modularity

Example usage:
python convert_waveglow.py \
    --src waveglow_256channels.pt \
    --dest model_weights/waveglow_256ch_state_dict.pt
"""


def main():
    parser = argparse.ArgumentParser()

    # Model Setup
    parser.add_argument("--src", default=None, type=str,
                        help="the source path to the full model file")
    parser.add_argument("--dst", default=None, type=str,
                        help="the destination path to save the state_dict")

    args = parser.parse_args()

    waveglow = torch.load(args.src)['model']
    torch.save(waveglow.state_dict(), args.dst)


if __name__ == "__main__":
    main()
