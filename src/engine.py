# --------------------------------------------------------------------------------
# Copyright (c) 2024 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
# 
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
# 
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------
"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import os
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from src.models import compute_metrics
import optuna


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.
    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).
    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    # Loop through data loader data batches
    for (X, y) in tqdm(dataloader, total=len(dataloader)):
        # Send data to target device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer/scheduler step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              slice_2d_majority_voting: bool = False) -> tuple[float, float, Tensor, Tensor]:
    """Tests a PyTorch model for a single epoch.
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.
    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    # Create empty tensors to store true and predicted labels
    y_true = torch.tensor([])
    y_pred = torch.tensor([])
    fold_results = []
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for (X, y) in tqdm(dataloader, total=len(dataloader)):
            # Send data to target device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # 0. Squeeze input if required
            if slice_2d_majority_voting:
                X = X.squeeze()
                y = y.repeat(X.shape[0])
            # 1. Forward pass
            test_pred_logits = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            if slice_2d_majority_voting:
                y = y[0].unsqueeze(0)
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            # Perform voting on 2D slices if parameter is set to True
            if slice_2d_majority_voting:
                counts = torch.bincount(test_pred_labels)
                test_pred_labels = torch.argmax(counts).unsqueeze(0)
            # Accumulate accuracy
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
            # Concatenate true and predicted labels
            y_true = torch.cat((y_true, y.cpu()), dim=0)
            y_pred = torch.cat((y_pred, test_pred_labels.cpu()), dim=0)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, y_true, y_pred


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          num_classes: int,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          warmup_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter,
          use_early_stop: bool = False,
          patience: int = 10,
          trial=None) -> Dict[str, List]:
    """Trains and validates a PyTorch model.
    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and validating the model
    in the same epoch loop.
    Calculates, prints and stores evaluation metrics throughout.
    Args:
    model: A PyTorch model to be trained and validated.
    train_dataloader: A DataLoader instance for the model to be trained on.
    val_dataloader: A DataLoader instance for the model to be validated on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    scheduler: A PyTorch scheduler to adjust the learning rate.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
    A dictionary of training and validating loss as well as training and
    validating accuracy metrics and other metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              val_loss: [...],
              val_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              val_loss: [1.2641, 1.5706],
              val_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": [],
               }
    max_valid_acc = 0
    max_valid_mcc = -1
    min_valid_loss = 1000
    epochs_no_improve = 0
    # Make sure model on target device
    model.to(device)
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        # Scheduler step
        if warmup_scheduler is not None and scheduler is not None:
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_scheduler.warmup_params[0]['warmup_period']:
                    scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        val_loss, val_acc, y_true, y_pred = test_step(model=model,
                                                      dataloader=val_dataloader,
                                                      loss_fn=loss_fn,
                                                      device=device)
        # Calculate and print metrics
        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
        # Print out what's happening
        print(f"\n\nEpoch: {epoch + 1}: ", end="\n\n")
        print(f"train_loss: {train_loss:.4f} ", end=" | ")
        print(f"train_acc: {train_acc:.4f} ", end=" | ")
        print(f"val_loss: {val_loss:.4f} ", end=" | ")
        for key in metrics.keys():
            print(f"{key[0].upper() + key[1:]}: {metrics[key]}", end=" | ")
        print("\n\n")
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        # Log to TensorBoard
        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss,
                                            "val_loss": val_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={"train_acc": train_acc,
                                            "val_acc": val_acc},
                           global_step=epoch)
        # early stopping
        if use_early_stop and metrics['mcc'] < max_valid_mcc:
            epochs_no_improve += 1
            print(f'EarlyStopping counter: {epochs_no_improve} out of {patience}')
            if epochs_no_improve == patience:
                print('Early stopping!')
                break
        # Save the best model according to validation MCC score, then validation accuracy, then validation loss
        if metrics['mcc'] > max_valid_mcc:
            # Save the best model according to validation MCC score
            print(f"\nSaving best model with val_mcc: {metrics['mcc']:.4f}\n")
            max_valid_mcc = metrics['mcc']
            max_valid_acc = metrics['accuracy']
            min_valid_loss = val_loss
            folder_path = os.path.join(writer.get_logdir(), 'torch_model')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(model.state_dict(), f'{folder_path}/best_model.pth')
            epochs_no_improve = 0
        elif metrics['mcc'] == max_valid_mcc:
            # If MCC is the same, save the model with the highest validation accuracy
            if metrics['accuracy'] > max_valid_acc:
                print(f"\nSaving best model with val_acc: {metrics['accuracy']:.4f}\n")
                max_valid_mcc = metrics['mcc']
                max_valid_acc = metrics['accuracy']
                min_valid_loss = val_loss
                folder_path = os.path.join(writer.get_logdir(), 'torch_model')
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(model.state_dict(), f'{folder_path}/best_model.pth')
            elif metrics['accuracy'] == max_valid_acc:
                # If accuracy is the same, save the model with the lowest validation loss
                if val_loss < min_valid_loss:
                    print(f"\nSaving best model with val_loss: {val_loss:.4f}\n")
                    min_valid_loss = val_loss
                    folder_path = os.path.join(writer.get_logdir(), 'torch_model')
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    torch.save(model.state_dict(), f'{folder_path}/best_model.pth')
        if trial is not None:
            trial.report(metrics['mcc'], epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
    # Close the writer
    writer.close()
    # Return the filled results at the end of the epochs
    return results
