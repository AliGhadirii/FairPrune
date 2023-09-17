import argparse
import yaml
import time
import os


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad

# from backpack import backpack, extend
# from backpack.extensions import DiagHessian

from Datasets.Fitz17k_dataset import get_fitz17k_dataloaders
from Models.Fitz17k_models import Fitz17kResNet18
from Utils.Misc_utils import set_seeds
from Evaluation import eval_model


# def get_parameter_salience(model_extend, metric_extend, batch, device):
#     inputs = batch["image"].to(device)
#     labels = batch["high"]
#     labels = torch.from_numpy(np.asarray(labels)).to(device)

#     output = model_extend(inputs.float())
#     loss = metric_extend(output, labels)
#     with backpack(DiagHessian()):
#         loss.backward()

#     return torch.cat([param.diag_h.flatten() for param in model_extend.parameters()])


def get_parameter_salience(model, metric, batch, device):
    inputs = batch["image"].to(device)
    labels = batch["high"]
    labels = torch.from_numpy(np.asarray(labels)).to(device)

    # Initialize the Hessian matrix
    hessian_matrix = torch.zeros(0).to(device)

    # Forward pass to calculate loss
    output = model(inputs.float())
    loss = metric(output, labels)
    averaged_loss = torch.mean(loss)

    # Calculate the Hessian diagonal for each parameter
    for param in model.parameters():
        grads = torch.autograd.grad(averaged_loss, param, create_graph=True)
        print("%%%%%%%%%%%%%%%")
        print(grads[0].shape)
        print("%%%%%%%%%%%%%%%")
        hessian_diag = torch.cat(
            [
                torch.autograd.grad(grad, param, retain_graph=True)[0].view(-1)
                for grad in grads
            ]
        )
        hessian_matrix = torch.cat((hessian_matrix, hessian_diag))

    return hessian_matrix


def fairprune(
    model,
    metric,
    device,
    config,
    verbose=False,
):
    """
    ARGS:
        model: model to be pruned
        metric: loss function to be used for saliency
        device: device to run on
        config: config file
        verbose: boolean check to print pruning information
    RETURNS:
        model: pruned model
    """
    # model_extend = extend(model).to(device)
    # metric_extend = extend(metric).to(device)

    dataloaders0, dataset_sizes0, num_classes0 = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        binary_subgroup=config["default"]["binary_subgroup"],
        fitz_filter=0,
        holdout_set="random_holdout",
        batch_size=config["FairPrune"]["batch_size"],
        num_workers=1,
    )

    dataloaders1, dataset_sizes1, num_classes1 = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        binary_subgroup=config["default"]["binary_subgroup"],
        fitz_filter=1,
        holdout_set="random_holdout",
        batch_size=config["FairPrune"]["batch_size"],
        num_workers=1,
    )

    # handling the compatibility of the given number of iterations with the dataloaders
    max_num_batches = max(len(dataloaders0["train"]), len(dataloaders1["train"]))
    if config["FairPrune"]["avg_num_batch"] > max_num_batches:
        raise ValueError(
            "The number of batches to calculate the average for should not exceed the maximum number of batches."
        )

    lengths_tensor = torch.tensor(
        [len(dataloaders0["train"]), len(dataloaders1["train"])]
    )
    min_length_index, min_length = torch.argmin(lengths_tensor), torch.min(
        lengths_tensor
    )

    all_saliencies = []
    train_iterator0 = iter(dataloaders0["train"])
    train_iterator1 = iter(dataloaders1["train"])

    iter_cnt = 0
    for batch0, batch1 in zip(train_iterator0, train_iterator1):
        # h0 = get_parameter_salience(model_extend, metric_extend, batch0, device)
        # h1 = get_parameter_salience(model_extend, metric_extend, batch1, device)

        # θ = torch.cat([param.flatten() for param in model_extend.parameters()])

        h0 = get_parameter_salience(model, metric, batch0, device)
        h1 = get_parameter_salience(model, metric, batch1, device)

        θ = torch.cat([param.flatten() for param in model.parameters()])

        print(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        )
        print(f"theta shape: {θ.shape}")
        print(f"h0 shape: {h0.shape}")
        print(f"h1 shape: {h1.shape}")
        print(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        )

        # saliency matrix
        saliency = 1 / 2 * θ**2 * (h0 - config["FairPrune"]["beta"] * h1)

        all_saliencies.append(saliency)

        # handling the smaller dataloader
        if iter_cnt % min_length == 0:
            if min_length_index == 0:
                train_iterator0 = iter(dataloaders0["train"])
            else:
                train_iterator1 = iter(dataloaders1["train"])

        # Breaking when the number of batches to calculate the average for is reached
        iter_cnt += 1
        if iter_cnt >= config["FairPrune"]["avg_num_batch"]:
            break

    # Stack the tensors in the list along a new dimension (0)
    all_saliencies = torch.stack(all_saliencies, dim=0)

    # Calculate the mean along the first dimension (0)
    average_saliency = torch.mean(all_saliencies, dim=0)

    k = int(
        config["FairPrune"]["prune_ratio"] * len(θ)
    )  # number of parameters to be pruned

    topk_indices = torch.topk(
        -average_saliency, k
    ).indices  # note we want to prune the smallest values hence negative

    # pruning the selected prameeters
    θ[topk_indices] = 0

    param_index = n_pruned = n_param = 0
    for name, param in model.named_parameters():
        # Note: bias is not pruned so explicitly avoiding
        if "bias" in name:
            continue
        num_params = param.numel()
        layer_saliency = θ[param_index : param_index + num_params].view(param.size())
        param.data = layer_saliency
        param_index += num_params

        if verbose:
            n_pruned += torch.sum(param.data == 0).item()
            n_param += num_params
            mean = round(torch.mean(layer_saliency).item(), 5)
            std = round(torch.std(layer_saliency).item(), 5)
            min_value = torch.min(layer_saliency).item()
            max_value = torch.max(layer_saliency).item()
            # n_positive_predictions = model(data[g0]).softmax(dim=1).argmax(axis=1).sum()
            print(
                "************************************************************************************************"
            )
            print(f"model parameter name: {name}")
            print(
                f"Pruned pram / total_params: {torch.sum(param == 0).item()} / {num_params}"
            )
            print(f"Statistics of the pruned prams in this parameter:")
            print(f"min: {min_value} / max: {max_value} / mean: {mean} / std: {std}")
            print(
                "************************************************************************************************"
            )

    if verbose:
        print(
            " --------------------------- Pruning Verification ---------------------------"
        )
        print(
            f"\nPruned {n_pruned} out of {n_param} parameters\n",
        )
        print(
            " ----------------------------------------------------------------------------"
        )

    return model


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        binary_subgroup=config["default"]["binary_subgroup"],
        holdout_set="random_holdout",
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    model = Fitz17kResNet18(
        num_classes=num_classes, pretrained=config["default"]["pretrained"]
    )
    model = model.to(device)

    best_BASE_model_path = os.path.join(
        config["output_folder_path"], "Resnet18_checkpoint_BASE.pth"
    )

    if os.path.isfile(best_BASE_model_path):
        print("Loading model from:", best_BASE_model_path)
        checkpoint = torch.load(best_BASE_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    metric = nn.CrossEntropyLoss(reduction="none")

    # Evaluating the BASE model
    val_metrics, _ = eval_model(
        model,
        dataloaders,
        dataset_sizes,
        device,
        config["default"]["level"],
        "BASE",
        config,
        save_preds=False,
    )

    best_bias_metric = val_metrics[config["FairPrune"]["target_bias_metric"]]
    prun_iter_cnt = 0
    no_improvement_cnt = 0

    while no_improvement_cnt < config["FairPrune"]["max_consecutive_no_improvement"]:
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt} +++++++++++++++++++++++++++++"
        )
        model_pruned = fairprune(
            model=model,
            metric=metric,
            device=device,
            config=config,
            verbose=True,
        )

        val_metrics, df_preds = eval_model(
            model_pruned,
            dataloaders,
            dataset_sizes,
            device,
            config["default"]["level"],
            "FairPrune",
            config,
            save_preds=False,
        )

        if val_metrics[config["FairPrune"]["target_bias_metric"]] > best_bias_metric:
            best_bias_metric = val_metrics[config["FairPrune"]["target_bias_metric"]]

            # Save the df_preds

            df_preds.to_csv(
                os.path.join(
                    config["output_folder_path"],
                    f"validation_results_Resnet18_FairPrune_Iter={prun_iter_cnt}.csv",
                ),
                index=False,
            )

            # Save the best model
            print("New leading model val metrics \n")
            print(val_metrics)

            best_model_path = os.path.join(
                config["output_folder_path"],
                f"Resnet18_checkpoint_FairPrune_Iter={prun_iter_cnt}.pth",
            )
            checkpoint = {
                "leading_val_metrics": val_metrics,
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, best_model_path)
            print("Checkpoint saved:", best_model_path)

            # Reset the counter
            consecutive_no_improvement = 0
        else:
            print("No improvement in the bias metric\n")
            consecutive_no_improvement += 1

        prun_iter_cnt += 1

        time_elapsed = time.time() - since
        print(
            "This iteration took {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
