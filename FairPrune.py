import argparse
import yaml
import time
import os
from tqdm import tqdm
import shutil
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact

from Datasets.dataloaders import get_dataloaders
from Models.Fitz17k_models import Fitz17kResNet18
from Utils.Misc_utils import set_seeds
from Utils.Metrics import plot_metrics
from Evaluation import eval_model


def get_parameter_salience(model_extend, metric_extend, batch, level, device):
    inputs = batch["image"].to(device)
    labels = batch[level]
    labels = torch.from_numpy(np.asarray(labels)).to(device)

    output = model_extend(inputs.float())
    loss = metric_extend(output, labels)

    with backpack(DiagGGNExact()):
        loss.backward()

    return torch.cat(
        [param.diag_ggn_exact.flatten().detach() for param in model_extend.parameters()]
    )


def fairprune(
    model,
    metric,
    device,
    config,
    verbose=1,
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

    model_extend = extend(model, use_converter=True)
    metric_extend = extend(metric.to(device))

    dataloaders0, dataset_sizes0, num_classes0 = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        level=config["default"]["level"],
        fitz_filter=0,
        batch_size=config["FairPrune"]["batch_size"],
        num_workers=1,
    )

    dataloaders1, dataset_sizes1, num_classes1 = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        level=config["default"]["level"],
        fitz_filter=1,
        batch_size=config["FairPrune"]["batch_size"],
        num_workers=1,
    )

    # handling the compatibility of the given number of iterations with the dataloaders
    max_num_batches = max(len(dataloaders0["train"]), len(dataloaders1["train"]))
    if config["FairPrune"]["num_batch_per_iter"] > max_num_batches:
        raise ValueError(
            "The number of batches to calculate the average for should not exceed the maximum number of batches."
        )

    lengths_tensor = torch.tensor(
        [len(dataloaders0["train"]), len(dataloaders1["train"])]
    )
    min_length_index = torch.argmin(lengths_tensor)

    train_iterator0 = iter(dataloaders0["train"])
    train_iterator1 = iter(dataloaders1["train"])

    # handling the smaller dataloader
    if min_length_index == 0:
        train_iterator0 = cycle(train_iterator0)
        print("INFO: Insufficient number of batches in dataloader0, cycling it.")
    else:
        train_iterator1 = cycle(train_iterator1)
        print("INFO: Insufficient number of batches in dataloader1, cycling it.")

    θ = torch.cat([param.flatten() for param in model_extend.parameters()])
    sum_saliencies = torch.zeros_like(θ)

    for iter_cnt, (batch0, batch1) in tqdm(
        enumerate(zip(train_iterator0, train_iterator1)),
        total=config["FairPrune"]["num_batch_per_iter"],
    ):
        h0 = get_parameter_salience(
            model_extend, metric_extend, batch0, config["default"]["level"], device
        )
        h1 = get_parameter_salience(
            model_extend, metric_extend, batch1, config["default"]["level"], device
        )

        # saliency matrix
        saliency = 1 / 2 * θ**2 * (h0 - config["FairPrune"]["beta"] * h1)

        sum_saliencies = sum_saliencies + saliency

        del h0, h1, saliency
        torch.cuda.empty_cache()

        # Breaking when the number of batches to calculate the average for is reached
        iter_cnt += 1
        if iter_cnt >= config["FairPrune"]["num_batch_per_iter"]:
            break

    # Calculate the mean along the first dimension (0)
    average_saliency = sum_saliencies / config["FairPrune"]["num_batch_per_iter"]

    k = int(
        config["FairPrune"]["pruning_rate"] * len(θ)
    )  # number of parameters to be pruned

    topk_indices = torch.topk(
        -average_saliency, k
    ).indices  # note we want to prune the smallest values hence negative

    mask = torch.ones(θ.shape).to(device)
    mask[topk_indices] = 0

    param_index = n_pruned = n_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()

        # Note: bias is not pruned so explicitly avoiding it
        if not "bias" in name:
            param.data = param.data * mask[param_index : param_index + num_params].view(
                param.size()
            )
            n_pruned += torch.sum(param.data == 0).item()

        param_index += num_params
        n_param += num_params

    if verbose > 0:
        print(
            " --------------------------- Pruning Verification ---------------------------"
        )
        print(
            f"\nPruned {n_pruned} out of {n_param} parameters\n",
        )
        print(
            " ----------------------------------------------------------------------------"
        )

    # get rid of the dataloaders to free up RAM
    dataloaders0 = None
    dataloaders1 = None

    # clear GPU cache to free up RAM
    torch.cuda.empty_cache()

    return model


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    shutil.copy(
        "Configs/configs_server.yml",
        os.path.join(config["output_folder_path"], "configs.yml"),
    )

    dataloaders, dataset_sizes, num_classes = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        level=config["default"]["level"],
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    model = (
        Fitz17kResNet18(
            num_classes=num_classes, pretrained=config["default"]["pretrained"]
        )
        .to(device)
        .eval()
    )

    checkpoint = torch.load(config["FairPrune"]["model_weights_path"])
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loading model weights from:", config["FairPrune"]["model_weights_path"])

    if num_classes == 2:
        metric = nn.BCEWithLogitsLoss()
    else:
        metric = nn.CrossEntropyLoss()

    prun_iter_cnt = 0
    consecutive_no_improvement = 0
    best_bias_metric = config["FairPrune"]["bias_metric_prev"]
    val_metrics_df = None

    while (
        consecutive_no_improvement
        <= config["FairPrune"]["max_consecutive_no_improvement"]
    ):
        since = time.time()

        print(
            f"+++++++++++++++++++++++++++++ Pruning Iteration {prun_iter_cnt+1} +++++++++++++++++++++++++++++"
        )
        if prun_iter_cnt == 0:
            pruned_model = fairprune(
                model=model,
                metric=metric,
                device=device,
                config=config,
                verbose=1,
            )
        else:
            pruned_model = fairprune(
                model=pruned_model,
                metric=metric,
                device=device,
                config=config,
                verbose=1,
            )

        model_name = f"ResNet18_FairPrune_PIter{prun_iter_cnt+1}"

        val_metrics, _ = eval_model(
            pruned_model,
            dataloaders,
            dataset_sizes,
            num_classes,
            device,
            config["default"]["level"],
            model_name,
            config,
            save_preds=True,
        )

        if config["FairPrune"]["target_bias_metric"] in [
            "AUC_Gap",
            "EOpp0",
            "EOpp1",
            "EOdd",
            "NAR",
        ]:
            if (
                val_metrics[config["FairPrune"]["target_bias_metric"]]
                < best_bias_metric
            ):
                best_bias_metric = val_metrics[
                    config["FairPrune"]["target_bias_metric"]
                ]

                # Save the best model
                print(
                    f'Achieved new leading val metrics: {config["FairPrune"]["target_bias_metric"]}={best_bias_metric} \n'
                )

                # Reset the counter
                consecutive_no_improvement = 0
            else:
                print(
                    f"No improvements observed in Iteration {prun_iter_cnt+1}, val metrics: \n"
                )
                consecutive_no_improvement += 1
        else:
            if (
                val_metrics[config["FairPrune"]["target_bias_metric"]]
                > best_bias_metric
            ):
                best_bias_metric = val_metrics[
                    config["FairPrune"]["target_bias_metric"]
                ]

                # Save the best model
                print(
                    f'Achieved new leading val metrics: {config["FairPrune"]["target_bias_metric"]}={best_bias_metric} \n'
                )

                # Reset the counter
                consecutive_no_improvement = 0
            else:
                print(
                    f"No improvements observed in Iteration {prun_iter_cnt+1}, val metrics: \n"
                )
                consecutive_no_improvement += 1

        print(val_metrics)

        model_path = os.path.join(
            config["output_folder_path"],
            f"{model_name}.pth",
        )
        checkpoint = {
            "config": config,
            "leading_val_metrics": val_metrics,
            "model_state_dict": pruned_model.state_dict(),
        }
        torch.save(checkpoint, model_path)

        prun_iter_cnt += 1

        time_elapsed = time.time() - since
        print(
            "This iteration took {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        if prun_iter_cnt == 0:
            val_metrics_df = pd.DataFrame([val_metrics])
        else:
            val_metrics_df = pd.concat(
                [val_metrics_df, pd.DataFrame([val_metrics])], ignore_index=True
            )
        val_metrics_df.to_csv(
            os.path.join(config["output_folder_path"], f"Pruning_metrics.csv"),
            index=False,
        )

        plot_metrics(val_metrics_df, ["accuracy", "acc_gap"], "ACC", config)
        plot_metrics(val_metrics_df, ["F1_Mac", "F1_Mac_gap"], "F1", config)
        plot_metrics(val_metrics_df, ["AUC", "AUC_Gap"], "AUC", config)
        plot_metrics(val_metrics_df, ["PQD", "DPM", "EOM"], "positive", config)
        plot_metrics(
            val_metrics_df,
            ["EOpp0", "EOpp1", "EOdd", "NAR", "NFR_Mac"],
            "negative",
            config,
        )

        plot_metrics(
            val_metrics_df,
            ["PQD_binary", "DPM_binary", "EOM_binary", "NAR_binary"],
            "binary",
            config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
