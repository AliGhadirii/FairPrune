import argparse
import yaml
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from sklearn.metrics import balanced_accuracy_score

from Datasets.Fitz17k_dataset import get_fitz17k_dataloaders
from Models.Fitz17k_models import Fitz17kResNet18
from Utils.Misc_utils import set_seeds


def train_model(
    dataloaders,
    dataset_sizes,
    num_classes,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
):
    since = time.time()
    batch_size = config["default"]["batch_size"]

    training_results = []
    validation_results = []
    start_epoch = 0

    best_model_path = os.path.join(
        config["default"]["output_folder_path"], "Resnet18_checkpoint_BASE.pth"
    )

    if best_model_path.is_file():
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

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            # Set the model to the training mode
            if phase == "train":
                model.train()
                scheduler.step()

            # Set model to the evaluation mode
            else:
                model.eval()

            # Running parameters
            running_loss = 0.0
            running_corrects = 0
            running_balanced_acc_sum = 0
            probs_mat = np.zeros((dataset_sizes[phase], num_classes))
            preds_vec = np.zeros((dataset_sizes[phase],))
            labels_vec = np.zeros((dataset_sizes[phase],))
            fitz = np.zeros(dataset_sizes[phase])
            cnt = 0

            print(f"Current phase: {phase}")
            # Iterate over data
            for batch in dataloaders[phase]:
                # Send inputs and labels to the device
                inputs = batch["image"].to(device)
                labels = batch["high"]
                attrs = batch["fitzpatrick"]

                labels = torch.from_numpy(np.asarray(labels)).to(device)
                attrs = torch.from_numpy(np.asarray(attrs)).to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.float()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    probs_mat[cnt * batch_size : (cnt + 1) * batch_size, :] = (
                        outputs.cpu().detach().numpy()
                    )
                    preds_vec[cnt * batch_size : (cnt + 1) * batch_size] = (
                        preds.cpu().detach().numpy()
                    )
                    labels_vec[cnt * batch_size : (cnt + 1) * batch_size] = (
                        labels.cpu().detach().numpy()
                    )
                    fitz[cnt * batch_size : (cnt + 1) * batch_size] = (
                        attrs.cpu().detach().numpy()
                    )

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
                    [phase, epoch, epoch_loss, epoch_acc, epoch_balanced_acc]
                )
            elif phase == "val":
                validation_results.append(
                    [phase, epoch, epoch_loss, epoch_acc, epoch_balanced_acc]
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
                }
                torch.save(checkpoint, best_model_path)
                print("Checkpoint saved:", best_model_path)

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


def eval_model(model, dataloaders, dataset_sizes, device, level, config):
    model = model.eval()
    prediction_list = []
    fitzpatrick_list = []
    hasher_list = []
    labels_list = []
    p_list = []
    topk_p = []
    topk_n = []
    d1 = []
    d2 = []
    d3 = []
    p1 = []
    p2 = []
    p3 = []
    with torch.no_grad():
        running_corrects = 0
        running_balanced_acc_sum = 0
        total = 0
        for batch in dataloaders["val"]:
            inputs = batch["image"].to(device)
            classes = batch[level]
            fitzpatrick = batch["fitzpatrick"]
            classes = torch.from_numpy(np.asarray(classes)).to(device)
            fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick)).to(device)
            hasher = batch["hasher"]

            outputs = model(inputs.float())  # (batchsize, classes num)
            probability = torch.nn.functional.softmax(outputs[0], dim=1)
            ppp, preds = torch.topk(probability, 1)  # topk values, topk indices
            if level == "low":
                _, preds5 = torch.topk(probability, 3)  # topk values, topk indices
                # topk_p.append(np.exp(_.cpu()).tolist())
                topk_p.append((_.cpu()).tolist())
                topk_n.append(preds5.cpu().tolist())
            running_corrects += torch.sum(preds.reshape(-1) == classes.data)
            running_balanced_acc_sum += (
                balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu())
                * inputs.shape[0]
            )
            p_list.append(ppp.cpu().tolist())
            prediction_list.append(preds.cpu().tolist())
            labels_list.append(classes.tolist())
            fitzpatrick_list.append(fitzpatrick.tolist())
            hasher_list.append(hasher)
            total += inputs.shape[0]
        acc = float(running_corrects) / float(dataset_sizes["val"])
        balanced_acc = float(running_balanced_acc_sum) / float(dataset_sizes["val"])

    if level == "low":
        for j in topk_n:  # each sample
            for i in j:  # in k
                d1.append(i[0])
                d2.append(i[1])
                d3.append(i[2])
        for j in topk_p:
            for i in j:
                # print(i)
                p1.append(i[0])
                p2.append(i[1])
                p3.append(i[2])

        def flatten(list_of_lists):
            if len(list_of_lists) == 0:
                return list_of_lists
            if isinstance(list_of_lists[0], list):
                return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
            return list_of_lists[:1] + flatten(list_of_lists[1:])

        df_x = pd.DataFrame(
            {
                "hasher": flatten(hasher_list),
                "label": flatten(labels_list),
                "fitzpatrick": flatten(fitzpatrick_list),
                "prediction_probability": flatten(p_list),
                "prediction": flatten(prediction_list),
                "d1": d1,
                "d2": d2,
                "d3": d3,
                "p1": p1,
                "p2": p2,
                "p3": p3,
            }
        )
    else:
        df_x = pd.DataFrame(
            {
                "hasher": flatten(hasher_list),
                "label": flatten(labels_list),
                "fitzpatrick": flatten(fitzpatrick_list),
                "prediction_probability": flatten(p_list),
                "prediction": flatten(prediction_list),
            }
        )

    num_epoch = config["default"]["n_epochs"]
    df_x.to_csv(
        os.path.join(
            config["default"]["output_folder_path"],
            f"validation_results_Resnet18_{num_epoch}_random_holdout_BASE.csv",
        ),
        index=False,
    )
    print(
        "\n Final Validation results: Accuracy: {}  Balanced Accuracy: {} \n".format(
            acc, balanced_acc
        )
    )


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        binary_subgroup=True,
        holdout_set="random_holdout",
        seed=42,
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    model = Fitz17kResNet18(
        num_classes=num_classes, pretrained=config["default"]["pretrained"]
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    model, training_results, validation_results = train_model(
        dataloaders,
        dataset_sizes,
        num_classes,
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        device,
        config,
    )

    num_epoch = config["default"]["n_epochs"]
    training_results.to_csv(
        os.path.join(
            config["default"]["output_folder_path"],
            f"training_results_Resnet18_{num_epoch}_random_holdout_BASE.csv",
        ),
        index=False,
    )
    validation_results.to_csv(
        os.path.join(
            config["default"]["output_folder_path"],
            f"all_validation_results_Resnet18_{num_epoch}_random_holdout_BASE.csv",
        ),
        index=False,
    )

    eval_model(
        model, dataloaders, dataset_sizes, device, config["default"]["level"], config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
