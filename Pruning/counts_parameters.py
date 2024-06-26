#this is for counting the weights 
#runs second

import argparse
import os

import torch

from emmental.modules import MaskedLinear

def expand_mask(mask, args):
    mask_block_rows = args.mask_block_rows
    mask_block_cols = args.mask_block_cols
    mask = torch.repeat_interleave(mask, mask_block_rows, dim=0)
    mask = torch.repeat_interleave(mask, mask_block_cols, dim=1)
    return mask


def main(args):
    serialization_dir = args.serialization_dir
    pruning_method = args.pruning_method
    threshold = args.threshold
    ampere_pruning_method = args.ampere_pruning_method

    st = torch.load(os.path.join(serialization_dir, "pytorch_model.bin"), map_location="cuda")

    remaining_count = 0  # Number of remaining (not pruned) params in the encoder
    encoder_count = 0  # Number of params in the encoder

    print("name".ljust(60, " "), "Remaining Weights %", "Remaining Weight")
    for name, param in st.items():
        if "encoder" not in name:
            continue

        if name.endswith(".weight"):
            weights = MaskedLinear.masked_weights_from_state_dict(st,
                                                                  name,
                                                                  pruning_method,
                                                                  threshold,
                                                                  ampere_pruning_method)
            mask_ones = (weights != 0).sum().item()
            print(name.ljust(60, " "), str(round(100 * mask_ones / param.numel(), 3)).ljust(20, " "), str(mask_ones))

            remaining_count += mask_ones
        elif MaskedLinear.check_name(name):
            pass
        else:
            encoder_count += param.numel()
            if name.endswith(".weight") and ".".join(name.split(".")[:-1] + ["mask_scores"]) in st:
                pass
            else:
                remaining_count += param.numel()

    print("")
    print("Remaining Weights (global) %: ", 100 * remaining_count / encoder_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruning_method",
        choices=["l0", "topK", "sigmoied_threshold"],
        type=str,
        required=True,
        help="Pruning Method (l0 = L0 regularization, topK = Movement pruning, sigmoied_threshold = Soft movement pruning)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="For `topK`, it is the level of remaining weights (in %) in the fine-pruned model."
        "For `sigmoied_threshold`, it is the threshold \tau against which the (sigmoied) scores are compared."
        "Not needed for `l0`",
    )
    parser.add_argument(
        "--serialization_dir",
        type=str,
        required=True,
        help="Folder containing the model that was previously fine-pruned",
    )
    parser.add_argument(
        "--mask_block_rows",
        default=1,
        type=int,
        help="Block row size for masks. Default is 1 -> general sparsity, not block sparsity.",
    )

    parser.add_argument(
        "--mask_block_cols",
        default=1,
        type=int,
        help="Block row size for masks. Default is 1 -> general sparsity, not block sparsity.",
    )
    args = parser.parse_args()

    main(args)
