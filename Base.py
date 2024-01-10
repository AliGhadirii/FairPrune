import argparse
import yaml
import time
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from sklearn.metrics import balanced_accuracy_score

from Datasets.dataloaders import get_dataloaders
from Models.Fitz17k_models import Fitz17kResNet18
from Utils.Misc_utils import set_seeds
from Evaluation import eval_model

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def train_model(
    dataloaders,
    dataset_sizes,
    num_classes,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    model_name,
    config,
):
    since = time.time()

    training_results = []
    validation_results = []
    start_epoch = 0
    best_acc = 0

    best_model_path = os.path.join(
        config["output_folder_path"], f"{model_name}_checkpoint_BASE.pth"
    )

    if os.path.isfile(best_model_path):
        print("Resuming training from:", best_model_path)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = checkpoint["best_loss"]
        best_acc = checkpoint["best_acc"]
        best_balanced_acc = checkpoint["best_balanced_acc"]
        leading_epoch = checkpoint["leading_epoch"]
        start_epoch = leading_epoch + 1

    for epoch in range(start_epoch, config["default"]["n_epochs"]):
        print("Epoch {}/{}".format(epoch, config["default"]["n_epochs"] - 1))
        print("-" * 20)

        since_epoch = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            # Set the model to the training mode
            if phase == "train":
                model.train()

            # Set model to the evaluation mode
            else:
                model.eval()

            # Running parameters
            running_loss = 0.0
            running_corrects = 0
            running_balanced_acc_sum = 0
            cnt = 0

            print(f"Current phase: {phase}")
            # Iterate over data
            for idx, batch in enumerate(dataloaders[phase]):
                # Send inputs and labels to the device
                inputs = batch["image"].to(device)
                labels = batch[config["default"]["level"]]

                if num_classes == 2:
                    labels = (
                        torch.from_numpy(np.asarray(labels)).unsqueeze(1).to(device)
                    )
                else:
                    labels = torch.from_numpy(np.asarray(labels)).to(device)

                # Zero the gradients
                optimizer.zero_grad()

                def handle_warning(
                    message, category, filename, lineno, file=None, line=None
                ):
                    print("Warning:", message)
                    print("Additional Information:")
                    print(f"Phase: {phase}, batch idx: {idx}")

                # Filter warnings to catch the specific warning types
                warnings.filterwarnings("always", category=UndefinedMetricWarning)

                # Set the custom function to handle the warning
                warnings.showwarning = handle_warning

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.float()
                    outputs = model(inputs)

                    if num_classes == 2:
                        probs = nn.functional.sigmoid(outputs)
                        theshold = find_threshold(
                            probs.cpu().data.numpy(), labels.cpu().data.numpy()
                        )
                        preds = (probs > theshold).to(torch.int32)

                        loss = criterion(outputs, labels.to(torch.float32))
                    else:
                        probs, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                    # Backward + optimize only if in the training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_balanced_acc_sum += balanced_accuracy_score(
                    labels.data.cpu(), preds.cpu()
                ) * inputs.size(0)

                # Increment
                cnt = cnt + 1

            if phase == "train":
                scheduler.step()

            # metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]

            # Print
            print(
                "{} Loss: {:.4f} Acc: {:.4f} Balanced Accuracy: {:.4f} ".format(
                    phase, epoch_loss, epoch_acc, epoch_balanced_acc
                )
            )

            # Save the accuracy and loss
            if phase == "train":
                training_results.append(
                    [phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc]
                )
            elif phase == "val":
                validation_results.append(
                    [phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc]
                )

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                print("New leading accuracy: {}".format(epoch_acc))
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_balanced_acc = epoch_balanced_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save checkpoint
                checkpoint = {
                    "leading_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                    "best_balanced_acc": best_balanced_acc,
                    "config": config,
                }
                torch.save(checkpoint, best_model_path)
                print("Checkpoint saved:", best_model_path)

        time_elapsed_epoch = time.time() - since_epoch
        print(
            "Epoch {} completed in {:.0f}m {:.0f}s".format(
                epoch, time_elapsed_epoch // 60, time_elapsed_epoch % 60
            )
        )

    # Time
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best val Balanced Acc: {:4f}".format(best_balanced_acc))

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    training_results = pd.DataFrame(training_results)
    training_results.columns = [
        "phase",
        "epoch",
        "loss",
        "accuracy",
        "balanced_accuracy",
    ]

    validation_results = pd.DataFrame(validation_results)
    validation_results.columns = [
        "phase",
        "epoch",
        "loss",
        "accuracy",
        "balanced_accuracy",
    ]

    return model, training_results, validation_results


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    model_name = f"Resnet18_BASE_{config['default']['level']}"

    dataloaders, dataset_sizes, num_classes = get_dataloaders(
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

    print(model)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), 0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    if config["default"]["mode"] == "eval":
        best_model_path = os.path.join(
            config["output_folder_path"], f"{model_name}_checkpoint_BASE.pth"
        )
        print("EVAL MODE: Loading the weights from ", best_model_path)

        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = checkpoint["best_loss"]
        best_acc = checkpoint["best_acc"]
        best_balanced_acc = checkpoint["best_balanced_acc"]
        leading_epoch = checkpoint["leading_epoch"]

        print(
            "Best epoch {}: Loss: {:.4f} Acc: {:.4f} Balanced Accuracy: {:.4f} ".format(
                leading_epoch, best_loss, best_acc, best_balanced_acc
            )
        )
    else:
        shutil.copy(
            "Configs/configs_server.yml",
            os.path.join(config["output_folder_path"], "configs.yml"),
        )

        model, training_results, validation_results = train_model(
            dataloaders,
            dataset_sizes,
            num_classes,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            model_name,
            config,
        )

        training_results.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"Training_log_{model_name}_random_holdout.csv",
            ),
            index=False,
        )
        validation_results.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"Validation_log_{model_name}_random_holdout.csv",
            ),
            index=False,
        )

    val_metrics, _ = eval_model(
        model,
        dataloaders,
        dataset_sizes,
        num_classes,
        device,
        config["default"]["level"],
        model_name,
        config,
        save_preds=True,
    )

    print("validation metrics:")
    print(val_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
