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
from src.data import get_transforms, get_dataloaders
from src.models import get_backbone, BackboneWithClassifier, AttentionalConv3D, TransformerConv3D, compute_metrics
from src.models.majority_voting_3d import MajorityVoting3D
from src.mynn import get_optim, get_schedul, get_warmup_schedul
from src.utils import set_seeds, plot_loss_curves
import torch
from torch import nn
import os


def run_experiment(train_df, val_df, test_df, config, writer, device, fold, trial=None):
    """
    Run the experiment for the specified fold this can be used also for a single fold.
    @param train_df: The training dataframe containing the paths to the images and the labels.
    @param val_df: The validation dataframe containing the paths to the images and the labels.
    @param test_df: The test dataframe containing the paths to the images and the labels.
    @param config: The configuration dictionary containing the parameters for the experiment.
    @param writer: The tensorboard writer for this experiment fold.
    @param device: The device to use for the experiment.
    @param fold: The fold number, used for the experiment log.
    @return: Tuple containing the true labels, the predicted labels and the metrics for this fold on the test set.
    """
    # Get the transformations and data augmentation strategy to apply to the images
    transform, data_augmentation, data_augmentation_slice = get_transforms(config['data_augmentation'],
                                                                           config["slicing_plane"],
                                                                           config["data_augmentation_slice"])
    print(f"Using {data_augmentation_slice} as data augmentation strategy on 2D slices\n")
    # Get the dataloaders for the training, validation and test sets
    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_slices=config['num_slices'],
        train_transform=transform,
        test_transform=transform,
        data_augmentation=data_augmentation,
        data_augmentation_slice=data_augmentation_slice,
        output_type=config['image_type'],
        revert_slices_order=config['revert_slices_order'],
        slicing_plane=config['slicing_plane'],
    )
    # Get the pretrained backbone on RadImageNet or ImageNet
    backbone, feat_map_dim = get_backbone(
        model_name=config['backbone'],
        radimgnet_classes=config['radimgnet_classes'],
        device=device,
        pretrained_on=config['pretrained_on']
    )
    # Set seeds for reproducibility
    set_seeds(config['random_seed'])
    # Get the model based on the image type and the 3D network
    if config['image_type'] == "2D":
        # Attach to the specified backbone a classifier for the majority voting case
        model = BackboneWithClassifier(
            backbone=backbone,
            embedding_dim=feat_map_dim,
            num_classes=num_classes,
            dropout=config['dropout']
        ).to(device)
        print(f"Using {config['backbone']} as backbone for 2D images with Majority Voting\n")
    elif config['image_type'] == "3D":
        if config['network_3D'] == "AttentionalConv3D":
            model = AttentionalConv3D(
                backbone=backbone,
                embedding_dim=feat_map_dim,
                num_classes=num_classes,
                num_slices=config['num_slices'],
                dropout=config['dropout']
            ).to(device)
            print(f"Using {config['backbone']} as backbone for AttentionalConv3D on 3D images\n")
        elif config['network_3D'] == "MajorityVoting3D":
            model = MajorityVoting3D(
                backbone=backbone,
                embedding_dim=feat_map_dim,
                num_slices=config['num_slices'],
                num_classes=num_classes,
                dropout=config['dropout']
            ).to(device)
            print(f"Using {config['backbone']} as backbone for MajorityVoting3D on 3D images\n")
        elif config['network_3D'] == "TransformerConv3D":
            model = TransformerConv3D(
                backbone=backbone,
                embedding_dim=feat_map_dim,
                num_slices=config['num_slices'],
                num_heads=config['num_heads'],
                num_classes=num_classes,
                dropout=config['dropout'],
                attn_dropout=config['attn_dropout'],
            ).to(device)
            print(f"Using {config['backbone']} as backbone for TransformerConv3DV2 on 3D images\n")
        else:
            raise ValueError("Invalid 3D network specified in config.yaml")
    else:
        raise ValueError("Invalid image type specified in config.yaml")
    # Load the pretrained model if specified
    if config['load_pretrained_model']:
        # Try to load the pretrained model
        try:
            model.load_state_dict(torch.load(config['pretrained_model_path'], map_location=device))
            model.classifier = nn.Sequential(
                nn.Dropout(config['dropout'], inplace=True),
                nn.Linear(feat_map_dim, num_classes)
            )
            model.to(device)
            print(f"Successfully loaded pretrained model from {config['pretrained_model_path']}\n")
        except FileNotFoundError:
            print(f"Pretrained model not found at {config['pretrained_model_path']}")
        except RuntimeError:
            print(f"Pretrained model at {config['pretrained_model_path']} has different architecture")
    # Resume the experiment if experiment folder exists
    if config['resume'] or config['skip_training']:
        print("Loading the best model for this fold and resume training...")
        best_model_path = os.path.join(config['experiment_folder'], f'fold_{fold}', 'torch_model',
                                       "best_model.pth")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    # Freeze the backbone with the specified percentage of layers
    num_layers = len(list(model.backbone.parameters()))
    num_layers_to_freeze = int(num_layers * config['freeze_first_percentage'])
    for i, param in enumerate(model.backbone.parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False
    print(f"Frozen first {num_layers_to_freeze} layers out of {num_layers} in the backbone\n")
    # Move the model to the device and parallelize it if specified
    if len(config['cuda_device']) > 1:
        model = nn.DataParallel(model, device_ids=config['cuda_device'])
        model.to(device)
    # Get the optimizer
    optimizer_class = get_optim(config['optimizer'])
    if config['optimizer_kwargs'] is not None:
        optimizer = optimizer_class(
            params=model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            **config['optimizer_kwargs']
        )
    else:
        optimizer = optimizer_class(
            params=model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    # Get the warmup scheduler if specified
    warmup_scheduler = None
    if config['warmup_scheduler'] is not None:
        warmup_scheduler = get_warmup_schedul(config["warmup_scheduler"], optimizer, config["warmup_kwargs"])
        print(f"Using {config['warmup_scheduler']} warmup scheduler with {config['warmup_kwargs']}\n")
    # Get the scheduler if specified
    scheduler = None
    if config['scheduler'] is not None:
        scheduler_class = get_schedul(config['scheduler'])
        print(type(scheduler_class))
        # Instantiate the scheduler with the keyword arguments specified in the config file
        if warmup_scheduler is not None:
            warmup_period = config["warmup_kwargs"]["warmup_period"]
            config['scheduler_kwargs']['T_max'] = config['num_epochs'] - warmup_period
        scheduler = scheduler_class(
            optimizer=optimizer,
            **config['scheduler_kwargs']
        )
        print(f"Using {config['scheduler']} scheduler with {config['scheduler_kwargs']}\n")
    if not config['skip_training']:
        results = engine.train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_classes=num_classes,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_scheduler=warmup_scheduler,
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
    else:
        print("Skipping the training phase...")
    # Get the best model for this fold and evaluate it on the test set
    print("Loading the best model for this fold and evaluate on test set...")
    if config['skip_training']:
        best_model_path = os.path.join(config['experiment_folder'], f"fold_{fold}", "torch_model", "best_model.pth")
    else:
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
