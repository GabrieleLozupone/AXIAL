# -----------------------------------------------------------------------------
# This file contains part of the project "Joint Learning Framework of Cross-Modal Synthesis
# and Diagnosis for Alzheimer's Disease by Mining Underlying Shared Modality Information".
# Original repository: https://github.com/thibault-wch/Joint-Learning-for-Alzheimer-disease.git
# -----------------------------------------------------------------------------

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
from src import engine
from src.models import compute_metrics
from src.utils import set_seeds, plot_loss_curves
import torch
from torch import nn
import os
from .get_dataloader import get_aware_loaders
from .networks3D import define_Cls
from torch import optim


def run_aware_experiment(train_df, val_df, test_df, config, writer, device, fold, trial=None):
    train_dataloader, val_dataloader, test_dataloader, num_classes = get_aware_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config
    )
    # Set seeds for reproducibility
    set_seeds(config['random_seed'])
    model = define_Cls(num_classes, "kaiming", 0.02, gpu_ids=config['cuda_device'])
    print("AwareNet initialized with kaiming method and gain 0.02")
    if config['load_pretrained_model']:
        try:
            model.load_state_dict(torch.load(config['pretrained_model_path'], map_location=device))
            model.to(device)
            print("Pretrained model loaded successfully")
        except FileNotFoundError:
            print("Pretrained model not found. Training from scratch...")
        except RuntimeError:
            print("Pretrained model incompatible with current model. Training from scratch...")
    if config['freeze_base']:
        # freeze percentage of base model
        num_layers = len(list(model.basic_module.parameters()))
        num_layers_to_freeze = int(num_layers * config['freeze_percentage'])
        for i, param in enumerate(model.basic_module.parameters()):
            if i < num_layers_to_freeze:
                param.requires_grad = False
        print(f"Freezing {config['freeze_base']}% of the base model")
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_classes=num_classes,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config['num_epochs'],
        device=device,
        writer=writer,
        trial=trial,
        use_early_stop=config['use_early_stopping'],
        patience=config['patience'],
    )
    # Plot loss and accuracy curves
    plt = plot_loss_curves(results=results)
    # Save the plot to the experiment folder
    plt.savefig(os.path.join(writer.get_logdir(), "loss_acc_curves.png"))
    # Get the best model for this fold and evaluate it on the test set
    print("Loading the best model for this fold and evaluate on test set...")
    best_model_path = os.path.join(writer.get_logdir(), 'torch_model', 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, _, y_true, y_pred = engine.test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        slice_2d_majority_voting=(config['image_type'] == "2D")
    )
    # Compute the metrics for this fold
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    print(f"\n\nTest results for fold {fold}: ", end="\n\n")
    print(f"Test loss: {test_loss:.4f} ", end=" | ")
    for key in metrics.keys():
        print(f"{key[0].upper() + key[1:]}: {metrics[key]}", end=" | ")
    # Save the model test results in with tensorboard
    writer.add_scalars(main_tag="Test results",
                       tag_scalar_dict=metrics,
                       global_step=fold)
    writer.close()
    return y_true, y_pred, metrics
