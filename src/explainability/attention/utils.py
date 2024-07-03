from src.data import train_val_test_subject_split
from src.data import load_dataframe
from src.models.aware_net.awarenet import AwareNet
from src.models.aware_net.get_dataloader import get_aware_loaders
from src.utils import get_device
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import os
from src.models import get_backbone
from src.data import get_transforms
from src.data import get_dataloaders
from src.models import Axial3D
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from torch.nn.functional import interpolate

yellow_light = "#FFF4E0"
yellow_dark = "#BDA77D"
blue_light = "#E0EDFF"
blue_dark = "#93AACC"
pink_light = "#FFE0F4"
pink_dark = "#BD7DA3"


def inference(model_name="Axial3DVGG16", random_seed=42, num_workers=10, cuda_idx=[3], num_slices=100):
    # Get device to test on
    device = get_device(cuda_idx=cuda_idx)
    task = ['CN', 'AD']
    dataset = "data/ADNI1_Complete_1_Yr_1.5T/ADNI_BIDS/dataset.csv"
    experiments_path = os.path.join("models/XAI", model_name)
    # Load the dataframe with all the subjects
    df, subjects = load_dataframe(dataset, task)
    # Define the K-Fold cross validation
    kf = KFold(n_splits=5,
               shuffle=True,
               random_state=random_seed)
    for i, (train_val_subj_index, test_subj_index) in enumerate(kf.split(subjects)):
        fold_num = i + 1
        print(f"------------------- Fold {fold_num} -------------------\n\n")
        # Get the test set for the current fold
        _, _, test_df = train_val_test_subject_split(df=df,
                                                     train_val_subj=subjects[train_val_subj_index],
                                                     test_subj=subjects[test_subj_index],
                                                     val_perc_split=0.2,
                                                     random_seed=random_seed)
        print(f"Number of subjects in the test set: {len(test_df)}")
        axial_plane_model = os.path.join(experiments_path, f"axial/fold_{fold_num}/best_model.pth")
        sagittal_plane_model = os.path.join(experiments_path, f"sagittal/fold_{fold_num}/best_model.pth")
        coronal_plane_model = os.path.join(experiments_path, f"coronal/fold_{fold_num}/best_model.pth")
        # Define the number of slices
        axial_num_slices = num_slices
        coronal_num_slices = num_slices
        sagittal_num_slices = num_slices
        # Get the pretrained backbone on ImageNet
        backbone, embedding_dim = get_backbone(
            model_name="VGG16",
            device=device,
            pretrained_on="ImageNet"
        )
        # Create a dict with the model paths and the number of slices with the name of the plane
        model_path_num_slices = {
            "axial": (axial_plane_model, axial_num_slices),
            "coronal": (coronal_plane_model, coronal_num_slices),
            "sagittal": (sagittal_plane_model, sagittal_num_slices)
        }
        # Create an empty dict to store the attention weights for each plane
        attention_weights = {
            "axial": torch.tensor([]),
            "coronal": torch.tensor([]),
            "sagittal": torch.tensor([])
        }
        pred_logits = {
            "axial": torch.tensor([]),
            "coronal": torch.tensor([]),
            "sagittal": torch.tensor([])
        }
        # Skip if the attention weights and predictions are already generated
        if os.path.exists(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "3D_attention_map.npy")):
            print("3D attention map already generated")
            continue
        for plane in model_path_num_slices:
            # Skip if the attention weights and predictions are already generated
            if os.path.exists(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}",
                                           f"{plane}/attention_weights.npy")):
                print(f"Attention weights and predictions already generated for {plane} plane")
                continue
            print("------------------- {} -------------------".format(plane))
            # Get the model path and the number of slices for the current plane
            model_path = model_path_num_slices[plane][0]
            num_slices = model_path_num_slices[plane][1]
            # Get the model path and the number of slices for the current plane
            transform, _, _ = get_transforms(slicing_plane=plane)
            # Number of classes
            num_classes = 2
            # Get the dataloader for the current plane
            if model_name == "Axial3DVGG16":
                _, _, dataloader, _ = get_dataloaders(train_df=test_df,
                                                      val_df=test_df,
                                                      test_df=test_df,
                                                      batch_size=2,
                                                      num_workers=num_workers,
                                                      num_slices=num_slices,
                                                      train_transform=transform,
                                                      test_transform=transform,
                                                      output_type="3D",
                                                      slicing_plane=plane)
                # Load the model
                model = Axial3D(backbone=backbone,
                                num_classes=num_classes,
                                embedding_dim=embedding_dim,
                                num_slices=model_path_num_slices[plane][1],
                                return_attention_weights=True
                                ).to(device)
            elif model_name == "AwareNet":
                _, _, dataloader, _ = get_aware_loaders(
                    train_df=test_df,
                    val_df=test_df,
                    test_df=test_df,
                    config={
                        "batch_size": 1,
                        "num_workers": num_workers,
                        "slicing_plane": plane,
                    }
                )
                model = AwareNet(num_classes=num_classes, return_attention_weights=True).to(device)
            else:
                raise ValueError("Model not found")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model for {plane} plane loaded")
            # Set the model to return attention weights
            model.return_attention_weights = True
            # Set the model to eval mode
            model.eval()
            with torch.inference_mode():
                for X, y in tqdm(dataloader, total=len(dataloader)):
                    # Move the data to the device
                    X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    # Forward pass
                    if model_name == "Axial3DVGG16":
                        preds_logs, weights = model(X)
                    elif model_name == "AwareNet":
                        preds_logs, _, weights, _, _ = model(X)
                    # Append the attention weights and the predictions
                    attention_weights[plane] = torch.cat((attention_weights[plane], weights.cpu()), dim=0)
                    pred_logits[plane] = torch.cat((pred_logits[plane], preds_logs.cpu()), dim=0)
            # Save the attention weights and the predictions
            path_to_save = os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", f"{plane}/")
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            # Save the attention weights
            np.save(os.path.join(path_to_save, "attention_weights.npy"), attention_weights[plane].numpy())
            # Save the predictions
            np.save(os.path.join(path_to_save, "predictions.npy"), pred_logits[plane].numpy())
            print(f"Attention weights and predictions saved for {plane} plane")
        # Load the attention weights for each plane
        attention_weights = {
            "axial": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "axial/attention_weights.npy"))),
            "coronal": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "coronal/attention_weights.npy"))),
            "sagittal": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "sagittal/attention_weights.npy")))
        }
        # Mean the attention weights over the subjects
        attention_weights["axial"] = attention_weights["axial"].mean(dim=0)
        attention_weights["coronal"] = attention_weights["coronal"].mean(dim=0)
        attention_weights["sagittal"] = attention_weights["sagittal"].mean(dim=0)
        print("Attention weights loaded", attention_weights["axial"].shape, attention_weights["coronal"].shape,
              attention_weights["sagittal"].shape)
        if model_name == "AwareNet":
            resizing_size = {
                "sagittal": 193,
                "axial": 193,
                "coronal": 229
            }
            for plane in attention_weights:
                # Interpolate the attention weights to the original size
                attention_weights[plane] = interpolate(attention_weights[plane].unsqueeze(0).unsqueeze(0),
                                                       size=(resizing_size[plane]), mode='linear',
                                                       align_corners=False).squeeze(0).squeeze(0)
            print("Attention weights resized for AwareNet")
        # Define a unique 3D attention map for the three planes
        attention_map = torch.zeros((attention_weights["sagittal"].shape[0],
                                     attention_weights["coronal"].shape[0],
                                     attention_weights["axial"].shape[0]))
        print("3D attention map created", attention_map.shape)
        # Fill the attention map with the attention weights of the three planes (mean)
        for i in tqdm(range(attention_map.shape[0])):
            for j in range(attention_map.shape[1]):
                for k in range(attention_map.shape[2]):
                    attention_map[i, j, k] = attention_weights["sagittal"][i] * attention_weights["coronal"][j] * \
                                             attention_weights["axial"][k]
        # normalize the attention map between 0 and 1
        attention_map -= attention_map.min()
        attention_map /= attention_map.max()
        # Print the attention map
        print(f"3D Attention Map generated shape: {attention_map.shape}")
        # Save the 3D attention map
        np.save(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "3D_attention_map.npy"),
                attention_map.cpu().numpy())


def save_histograms_folds(model_name="Axial3DVGG16", percentile=75, pad=False, tollerance=2):
    # Plot the histograms for the attention weights for each fold
    for fold_num in range(1, 6):
        # Load the attention weights for each plane
        attention_weights = {
            "axial": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "axial/attention_weights.npy"))),
            "coronal": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "coronal/attention_weights.npy"))),
            "sagittal": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "sagittal/attention_weights.npy")))
        }
        # Mean the attention weights over the subjects
        attention_weights["axial"] = attention_weights["axial"].mean(dim=0).unsqueeze(0)
        attention_weights["coronal"] = attention_weights["coronal"].mean(dim=0).unsqueeze(0)
        attention_weights["sagittal"] = attention_weights["sagittal"].mean(dim=0).unsqueeze(0)
        for plane in attention_weights:
            # Load the numpy array from the .npy file
            plane_weights = attention_weights[plane].cpu().numpy().flatten()
            plot_histogram(plane_weights, plane, model_name=model_name, fold_num=fold_num, percentile=percentile,
                           tollerance=tollerance)


def plot_histogram(plane_weights, plane, model_name, fold_num, percentile, tollerance):
    # Define the colors
    main_color = blue_light
    secondary_color = blue_dark
    third_color = pink_dark
    fourth_color = pink_light
    # Plot the histogram
    fig, ax = plt.subplots(figsize=(20, 7))
    # Update the ticks fontsize
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.bar(range(0, len(plane_weights)), plane_weights, width=1.0, color=main_color, label='Attention Score')
    ax.set_xlabel('Slice Number', fontsize=18)
    ax.set_ylabel('Attention Score', fontsize=18)
    ax.set_title(f'{plane.capitalize()} Slice Attention Distribution', fontsize=20)
    # Setting the x-axis and y-axis ranges
    ax.set_xlim(-1, len(plane_weights))  # Adjust as needed
    ax.set_ylim(0, max(plane_weights) * 1.1)  # Adjust as needed to ensure visibility of all data points
    # Add arrowheads
    ax.plot(0, 1, '^k', transform=ax.transAxes, clip_on=False)
    ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)
    # Highlight the ranges with the highest attention scores, defined as scores above the 90th percentile
    high_attention_threshold = np.percentile(plane_weights, percentile)
    for i, score in enumerate(plane_weights):
        if score >= high_attention_threshold:
            ax.bar(i, score, width=1.0, color=secondary_color)
    # Find the continuous ranges of slices with the highest attention scores (some histogram are bimodal)
    intervals = []
    for i, score in enumerate(plane_weights):
        if score >= high_attention_threshold:
            if not intervals:
                intervals.append([i, i])
            else:
                if i == intervals[-1][1] + 1:
                    intervals[-1][1] = i
                else:
                    intervals.append([i, i])
    # Remove ranges with less than tollerance slices
    intervals = [interval for interval in intervals if interval[1] - interval[0] > tollerance]
    for interval in intervals:
        start_interval = interval[0]
        end_interval = interval[1]
        ax.axvline(x=start_interval - 0.5, color=third_color, linestyle='--')
        ax.axvline(x=end_interval + 0.5, color=third_color, linestyle='--')
        string_start = str(int(start_interval - 0.5)) if start_interval > 10 else " " + str(
            int(start_interval - 0.5)) + " "
        string_end = str(int(end_interval + 0.5)) if end_interval > 10 else " " + str(int(end_interval + 0.5)) + " "
        start_text = ax.text(start_interval - 0.5, 0, string_start, color=third_color, ha='center', fontsize=20)
        end_text = ax.text(end_interval + 0.5, 0, string_end, color=third_color, ha='center', fontsize=20)
        start_text.set_bbox(dict(facecolor=fourth_color, alpha=1, edgecolor=third_color, boxstyle='round,pad=0.5'))
        end_text.set_bbox(dict(facecolor=fourth_color, alpha=1, edgecolor=third_color, boxstyle='round,pad=0.5'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Save the histogram
    os.makedirs(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", f"{plane}"), exist_ok=True)
    plt.savefig(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", f"{plane}",
                             f"{plane}_attention_histogram.png"), bbox_inches='tight')
    plt.savefig(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", f"{plane}",
                             f"{plane}_attention_histogram.pdf"), bbox_inches='tight')


def entire_dataset_histogram(model_name="Axial3DVGG16", percentile=75, pad=False, tollerance=2):
    attention_weights = {
        "axial": torch.tensor([]),
        "coronal": torch.tensor([]),
        "sagittal": torch.tensor([])
    }
    for fold_num in range(1, 6):
        # Load the attention weights for each plane
        fold_attention_weights = {
            "axial": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "axial/attention_weights.npy"))),
            "coronal": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "coronal/attention_weights.npy"))),
            "sagittal": torch.tensor(np.load(
                os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "sagittal/attention_weights.npy")))
        }
        # Mean the attention weights over the subjects
        fold_attention_weights["axial"] = fold_attention_weights["axial"].mean(dim=0).unsqueeze(0)
        fold_attention_weights["coronal"] = fold_attention_weights["coronal"].mean(dim=0).unsqueeze(0)
        fold_attention_weights["sagittal"] = fold_attention_weights["sagittal"].mean(dim=0).unsqueeze(0)
        # Concatenate the attention weights for each plane
        attention_weights["axial"] = torch.cat((attention_weights["axial"], fold_attention_weights["axial"]), dim=0)
        attention_weights["coronal"] = torch.cat((attention_weights["coronal"], fold_attention_weights["coronal"]),
                                                 dim=0)
        attention_weights["sagittal"] = torch.cat((attention_weights["sagittal"], fold_attention_weights["sagittal"]),
                                                  dim=0)
    # Mean the attention weights over the folds
    attention_weights["axial"] = attention_weights["axial"].mean(dim=0).unsqueeze(0)
    attention_weights["coronal"] = attention_weights["coronal"].mean(dim=0).unsqueeze(0)
    attention_weights["sagittal"] = attention_weights["sagittal"].mean(dim=0).unsqueeze(0)
    if model_name == "AwareNet":
        resizing_size = {
            "sagittal": 193,
            "axial": 193,
            "coronal": 229
        }
        for plane in attention_weights:
            # Interpolate the attention weights to the original size
            attention_weights[plane] = interpolate(attention_weights[plane].unsqueeze(0), size=(resizing_size[plane]),
                                                   mode='linear', align_corners=False).squeeze(0)
        print("Attention weights resized for AwareNet")
    fold_num = "entire_dataset"
    if not os.path.exists(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "3D_attention_map.npy")):
        # Define a unique 3D attention map for the three planes
        attention_map = torch.zeros((attention_weights["sagittal"].shape[1],
                                     attention_weights["coronal"].shape[1],
                                     attention_weights["axial"].shape[1]))
        print("3D attention map created", attention_map.shape)
        # Fill the attention map with the attention weights of the three planes (mean)
        for i in tqdm(range(attention_map.shape[0])):
            for j in range(attention_map.shape[1]):
                for k in range(attention_map.shape[2]):
                    attention_map[i, j, k] = attention_weights["sagittal"][0][i] * attention_weights["coronal"][0][j] * \
                                             attention_weights["axial"][0][k]
        # normalize the attention map between 0 and 1
        attention_map -= attention_map.min()
        attention_map /= attention_map.max()
        # Print the attention map
        print(f"3D Attention Map for entire dataset generated. Shape: {attention_map.shape}")
        # Save the 3D attention map
        os.makedirs(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}"), exist_ok=True)
        np.save(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", "3D_attention_map.npy"),
                attention_map.cpu().numpy())
    else:
        print("3D attention map for entire dataset already generated")
    # Plot the histogram for the entire dataset
    for plane in attention_weights:
        num_total_slices = 193
        # Load the numpy array from the .npy file
        plane_weights = attention_weights[plane].cpu().numpy().flatten()
        if pad == True:
            padding = (int(np.ceil((num_total_slices - len(plane_weights)) / 2.0)),
                       int(np.floor((num_total_slices - len(plane_weights)) / 2.0)))
            plane_weights = np.pad(plane_weights, padding, mode='constant', constant_values=0)
        plot_histogram(plane_weights, plane, model_name=model_name, fold_num="entire_dataset", percentile=percentile,
                       tollerance=tollerance)


def generate_explainable_mri(model_name="Axial3DVGG16", fold_num="entire_dataset", amplification_factor=10):
    try:
        attention_map = np.load(f"explainability/{model_name}/fold_{fold_num}/3D_attention_map.npy")
    except:
        print("3D attention map not found")
        return
    print(f"3D Attention Map loaded. Shape: {attention_map.shape}")
    # Load the template image to overlay the attention map
    template_mri = nib.load("template/mni_icbm152_t1_tal_nlin_sym_09c.nii")
    template_data = template_mri.get_fdata()
    # Define the padding to apply to obtain the same shape as the template image
    padding = [(int(np.ceil((m - t) / 2.0)), int(np.floor((m - t) / 2.0))) for m, t in
               zip(template_data.shape, attention_map.shape)]
    # Apply the padding
    attention_map = np.pad(attention_map, padding, mode='constant', constant_values=0)
    # Normalize the template image
    template_data -= template_data.min()
    template_data /= template_data.max()
    # Apply the attention map with amplification factor
    explainable_mri = template_data + amplification_factor * attention_map
    # Save the explainable MRI
    explainable_mri_nii = nib.Nifti1Image(explainable_mri, template_mri.affine, template_mri.header)
    nib.save(explainable_mri_nii, f"explainability/{model_name}/fold_{fold_num}/explainable_mri.nii")
    print("Explainable MRI saved")


def compute_xai_metrics(model_name="Axial3DVGG16", fold_num="entire_dataset", percentile=99.9):
    try:
        attention_map = np.load(f"explainability/{model_name}/fold_{fold_num}/3D_attention_map.npy")
    except:
        print("3D attention map not found")
        return
    print(f"3D Attention Map loaded. Shape: {attention_map.shape}")
    try:
        # Load the atlas
        atlas = nib.load("template/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii")
        atlas_data = atlas.get_fdata()
    except:
        print("Atlas not found")
        return
    print(f"Atlas loaded. Shape: {atlas.shape}")
    # Define the padding to apply to obtain the same shape as the template image
    padding = [(int(np.ceil((m - t) / 2.0)), int(np.floor((m - t) / 2.0))) for m, t in
               zip(atlas_data.shape, attention_map.shape)]
    # Apply the padding
    attention_map = np.pad(attention_map, padding, mode='constant', constant_values=0)
    # Compute the binarized attention map
    binary_map = attention_map > np.percentile(attention_map, percentile)
    # Read the csv file with the labels of the regions
    regions = pd.read_csv("template/CerebrA_LabelDetails.csv")
    mapping = {row['Label Name']: {'right': row['RH Label'], 'left': row['LH Labels']} for _, row in regions.iterrows()}
    # Prepare a list to hold all the formatted data
    data = []
    # Calculate overlap information for each region in the atlas
    for region_id in np.unique(atlas_data):
        if region_id == 0:  # Assuming 0 is background or non-interest region
            continue
        # Find voxels that overlap between the binary heatmap and the current atlas region
        overlap = (binary_map > 0) & (atlas_data == region_id)
        overlap_count = np.sum(overlap)
        if overlap_count > 0:
            # Calculate mean, standard deviation, maximum, and minimum values in the attention map for the overlapped region
            overlap_mean = np.mean(attention_map[overlap])
            overlap_std = np.std(attention_map[overlap])
            overlap_max = np.max(attention_map[overlap])
            overlap_min = np.min(attention_map[overlap])
            # Calculate the percentage overlap with respect to the atlas region
            overlap_percentage_region = overlap_count / np.sum(atlas_data == region_id)
            # Determine the corresponding brain area and hemisphere
            for region, ids in mapping.items():
                if region_id in [ids['right'], ids['left']]:
                    hemisphere = 'right' if ids['right'] == region_id else 'left'
                    brain_area = f"{region} - {hemisphere}"
                    # Store all computed values in the data list
                    data.append([
                        brain_area,
                        overlap_count,
                        overlap_mean,
                        overlap_std,
                        overlap_max,
                        overlap_min,
                        overlap_percentage_region
                    ])
                    break
    # Creating DataFrame with specified columns
    df = pd.DataFrame(data, columns=["Brain Area", "Volume", "Attention Mean", "Attention STD", "Attention Max",
                                     "Attention Min", "Percentage of region"])
    df.sort_values(by="Volume", ascending=False, inplace=True)
    print("Dataframe created\n", df)
    df.to_csv(f"explainability/{model_name}/fold_{fold_num}/xai_metrics.csv", index=False)
    print("XAI metrics saved")

