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
import os
import cv2
import nibabel as nib
import pandas as pd
import torch
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)
import numpy as np
from sklearn.model_selection import KFold
from torchvision import transforms
from tqdm.auto import tqdm
from src.data import get_dataloaders
from src.data import get_transforms
from src.data import load_dataframe
from src.data import train_val_test_subject_split
from src.models import Axial3D, TransformerConv3D
from src.models import get_backbone
from src.models.aware_net.awarenet import AwareNet
from src.models.aware_net.get_dataloader import get_aware_loaders
from src.utils import get_device
import matplotlib.pyplot as plt

methods = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "gradcamelementwise": GradCAMElementWise
}


def generate_3d_cam(cams, original_shapes, num_slices):
    """
    Generate 3D CAM from 2D CAMs from each plane.
    :param cams: 2D CAMs from each plane
    :param original_shapes: original shape of the image
    :param num_slices: number of slices in each plane
    :return: 3D CAM
    """
    tmp_matrixes = {
        "axial": np.zeros(original_shapes),
        "coronal": np.zeros(original_shapes),
        "sagittal": np.zeros(original_shapes),
    }
    for plane in cams:
        for i, cam in enumerate(cams[plane]):
            if plane == "sagittal":
                tmp_matrixes[plane][original_shapes[0] // 2 - num_slices[plane] // 2 + i, :, :] = cam
            elif plane == "coronal":
                tmp_matrixes[plane][:, original_shapes[1] // 2 - num_slices[plane] // 2 + i, :] = cam
            else:
                tmp_matrixes[plane][:, :, original_shapes[2] // 2 - num_slices[plane] // 2 + i] = cam
    print(f"axial: {tmp_matrixes['axial'].shape}")
    print(f"coronal: {tmp_matrixes['coronal'].shape}")
    print(f"sagittal: {tmp_matrixes['sagittal'].shape}")
    cam_3d_map = np.zeros(original_shapes)
    for i in range(original_shapes[0]):
        for j in range(original_shapes[1]):
            for k in range(original_shapes[2]):
                cam_3d_map[i, j, k] = tmp_matrixes["axial"][i, j, k] * tmp_matrixes["coronal"][i, j, k] * \
                                      tmp_matrixes["sagittal"][i, j, k]
    print(f"3D cam generated shape: {cam_3d_map.shape}")
    # normalize the cam
    cam_3d_map = cam_3d_map - cam_3d_map.min()
    cam_3d_map = cam_3d_map / cam_3d_map.max()
    print(f"3D cam normalized")
    return cam_3d_map


def save_cams(model_name="TransformerConv3DV2", cam_method="gradcam++", num_workers=20, cuda_idxs=[0], num_slices=100,
              random_seed=42):
    """
    Save the CAMs for each fold of the dataset for the given model and cam method.
    :param model_name: name of the model
    :param cam_method: method to compute the CAM
    :param num_workers: number of workers for the dataloader
    :param cuda_idxs: list of cuda indexes
    :param num_slices: number of slices to consider
    :param random_seed: random seed for the KFold
    """
    # Get device to test on
    device = get_device(cuda_idx=cuda_idxs)
    task = ['CN', 'AD']
    dataset = "data/ADNI_BIDS/dataset.csv"
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
        # Define the paths to the best models for each plane
        axial_plane_model = os.path.join(experiments_path, f"axial/fold_{fold_num}/best_model.pth")
        sagittal_plane_model = os.path.join(experiments_path, f"sagittal/fold_{fold_num}/best_model.pth")
        coronal_plane_model = os.path.join(experiments_path, f"coronal/fold_{fold_num}/best_model.pth")
        cam_tensors = {
            "axial": [],
            "coronal": [],
            "sagittal": []
        }
        # Define the number of slices
        axial_num_slices = num_slices
        coronal_num_slices = num_slices
        sagittal_num_slices = num_slices
        if model_name != "AwareNet":
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
        for plane in model_path_num_slices:
            model_path = model_path_num_slices[plane][0]
            num_slices = model_path_num_slices[plane][1]
            # Skip if file .npy already exists
            if os.path.exists(
                    os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", f"{plane}/{cam_method}.npy")):
                print(f"File {cam_method}.npy already exists for {model_name} in {plane} plane")
                continue
            # Get the model path and the number of slices for the current plane
            transform, _, _ = get_transforms(slicing_plane=plane)
            simple_transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
            ])
            # Get the dataloader for the current plane
            if model_name == "AwareNet":
                _, _, dataloader, num_classes = get_aware_loaders(
                    train_df=test_df,
                    val_df=test_df,
                    test_df=test_df,
                    config={
                        "batch_size": 1,
                        "num_workers": num_workers,
                        "slicing_plane": plane,
                    }
                )
                model = AwareNet(num_classes=num_classes, return_attention_weights=False).to(device)
            else:
                _, _, dataloader, num_classes = get_dataloaders(train_df=test_df,
                                                                val_df=test_df,
                                                                test_df=test_df,
                                                                batch_size=1,
                                                                num_workers=num_workers,
                                                                num_slices=num_slices,
                                                                train_transform=transform,
                                                                test_transform=simple_transform,
                                                                output_type="3D",
                                                                slicing_plane=plane)
                if model_name == "TransformerConv3D":
                    model = TransformerConv3D(
                        backbone=backbone,
                        embedding_dim=embedding_dim,
                        num_slices=num_slices,
                        num_heads=4,
                        num_classes=num_classes,
                        dropout=0.0,
                        attn_dropout=0.0,
                    ).to(device)
                elif model_name == "Axial3DVGG16":
                    model = Axial3D(backbone=backbone,
                                    num_classes=num_classes,
                                    embedding_dim=embedding_dim,
                                    num_slices=num_slices,
                                    return_attention_weights=False
                                    ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model {model_name} for {plane} plane loaded")
            for input, _ in tqdm(dataloader, total=len(dataloader)):
                # Move the data to the device
                input_tensor = input.to(device, non_blocking=True)
                for i in range(input_tensor.shape[1]):
                    if model_name != "AwareNet":
                        input_tensor[:, i, :, :, :] = transform(input[:, i, :, :, :])
                # Define the CAM method
                cam_alghoritm = methods[cam_method]
                if model_name == "AwareNet":
                    target_layers = [model.att_module.loc_att.module[0]]
                else:
                    target_layers = [model.backbone[-1]]
                with cam_alghoritm(model=model,
                                   target_layers=target_layers) as cam:
                    # Get the CAMs
                    cam_tensors[plane].append(cam(input_tensor, targets=None))
            # Save the CAMs in numpy format
            path_to_save = os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", f"{plane}/")
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            np.save(os.path.join(path_to_save, f"{cam_method}.npy"), cam_tensors[plane])
            print(f"Saved CAMs for {model_name} in {plane} plane")


def plot_cams(visualization_cams, model_name="TransformerConv3D", fold_num=1, cam_method="gradcam++"):
    """
    Plot the CAMs for each plane and save the visualization in a png file.
    :param visualization_cams: dictionary with the visualization of the CAMs for each plane
    :param model_name: name of the model
    :param fold_num: number of the fold
    :param cam_method: method used to compute the CAM
    """
    for plane in visualization_cams:
        fig, axs = plt.subplots(len(visualization_cams[plane]) // 10, 10, figsize=(20, 15))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i, ax in enumerate(axs.flatten()):
            im = ax.imshow(np.rot90(visualization_cams[plane][i], k=1),
                           aspect='auto')  # Ensure consistent scaling across all plots
            ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"explainability/{model_name}/fold_{fold_num}/{plane}/visualization_{cam_method}.png")


def generate_explainable_mri(map_3d, template, amplification_factor=10.0):
    """
    Generate the explainable MRI from the 3D CAM, the cam is amplified by a factor and overlapped with the template.
    :param map_3d: 3D CAM
    :param template: template MRI
    :param amplification_factor: amplification factor
    :return: explainable MRI
    """
    template_data = template.get_fdata()
    template_data = template_data - template_data.min()
    template_data = template_data / template_data.max()
    explainable_mri = template_data + map_3d * amplification_factor
    return nib.Nifti1Image(explainable_mri, template.affine, template.header)


def save_mean_cams_folds(model_name="TransformerConv3D", cam_method="gradcam++"):
    # Load the template
    template = nib.load("template/mni_icbm152_t1_tal_nlin_sym_09c.nii")
    template_data = template.get_fdata()
    print(f"Template loaded with shape:{template_data.shape}")
    # Resize the images according to the original mri dimension
    original_shapes = {
        "axial": (229, 193),
        "coronal": (193, 193),
        "sagittal": (193, 229)
    }
    num_slices = {
        "axial": 100,
        "coronal": 100,
        "sagittal": 100,
    }
    for fold_num in range(1, 6):
        visualization_cams = {
            "axial": [],
            "coronal": [],
            "sagittal": []
        }
        resized_grayscale_cams = {
            "axial": [],
            "coronal": [],
            "sagittal": []
        }
        # Load the cams for the the plane
        for plane in ["axial", "coronal", "sagittal"]:
            cams = np.load(f"explainability/{model_name}/fold_{fold_num}/{plane}/{cam_method}.npy")
            print(f"Fold {fold_num} - {plane} - {cams.shape} loaded")
            # Compute the mean over the subjects
            mean_cam = np.mean(cams, axis=0)
            for i, cam in enumerate(mean_cam):
                if model_name == "AwareNet":
                    cam_rot = mean_cam[i, :, :]
                    s1 = 128 - original_shapes[plane][0] // 2 - original_shapes[plane][0] % 2
                    s2 = s1 + original_shapes[plane][0]
                    s3 = 128 - original_shapes[plane][1] // 2 - original_shapes[plane][1] % 2
                    s4 = s3 + original_shapes[plane][1]
                    cam_rot = cam_rot[s1:s2, s3:s4]
                    cam_rot = np.rot90(cam_rot, k=1, axes=(1, 0))
                    if plane == "sagittal":
                        s1 = (128 - template_data.shape[0] // 2 - template_data.shape[0] % 2)
                        s2 = s1 + template_data.shape[0]
                    elif plane == "coronal":
                        s1 = (128 - template_data.shape[1] // 2 - template_data.shape[1] % 2)
                        s2 = s1 + template_data.shape[1]
                    elif plane == "axial":
                        s1 = (128 - template_data.shape[2] // 2 - template_data.shape[2] % 2)
                        s2 = s1 + template_data.shape[2]
                    if i >= s1 and i < s2:
                        resized_grayscale_cams[plane].append(cam_rot)
                else:
                    cam_rot = np.rot90(mean_cam[i, :, :], k=1, axes=(1, 0))
                    resized_grayscale_cams[plane].append(cv2.resize(cam_rot, original_shapes[plane]))
            print(
                f"Fold {fold_num} - {plane} - resized grayscale cams generated with shape: {len(resized_grayscale_cams[plane]), resized_grayscale_cams[plane][0].shape}")
            for i in range(0, num_slices[plane]):
                # Get the centered N slices for each plane
                if plane == "sagittal":
                    index = template_data.shape[0] // 2 - num_slices[plane] // 2 + i
                    template_slice = template_data[index, :, :]
                elif plane == "coronal":
                    index = template_data.shape[1] // 2 - num_slices[plane] // 2 + i
                    template_slice = template_data[:, index, :]
                else:
                    index = template_data.shape[2] // 2 - num_slices[plane] // 2 + i
                    template_slice = template_data[:, :, index]
                # Normalize the image between 0 and 1
                template_slice = template_slice - template_slice.min()
                template_slice = template_slice / template_slice.max()
                template_slice = np.float32(template_slice)
                # Convert to RGB
                template_slice = cv2.cvtColor(template_slice, cv2.COLOR_GRAY2RGB)
                # Get the cam based on method
                if model_name == "AwareNet":
                    cam_to_add = resized_grayscale_cams[plane][index]
                else:
                    cam_to_add = resized_grayscale_cams[plane][i]
                cam_result = show_cam_on_image(template_slice, cam_to_add, use_rgb=True, colormap=cv2.COLORMAP_JET)
                visualization_cams[plane].append(cam_result)
            print(
                f"Fold {fold_num} - {plane} - visualization generated with shape: {len(visualization_cams[plane]), visualization_cams[plane][0].shape}")
            print(
                f"Fold {fold_num} - {plane} - resized grayscale cams generated with shape: {len(resized_grayscale_cams[plane]), resized_grayscale_cams[plane][0].shape}")
        # Save the cam result
        plot_cams(visualization_cams=visualization_cams, model_name=model_name, fold_num=fold_num,
                  cam_method=cam_method)
        cam_3d_map = generate_3d_cam(resized_grayscale_cams, template_data.shape, num_slices)
        # Save the 3D cam
        np.save(f"explainability/{model_name}/fold_{fold_num}/3D_CAM_{cam_method}.npy", cam_3d_map)
        # Generate the explainable MRI
        explainable_mri = generate_explainable_mri(cam_3d_map, template)
        # Save the explainable MRI
        nib.save(explainable_mri, f"explainability/{model_name}/fold_{fold_num}/explainable_mri_{cam_method}.nii")


def entire_dataset_cams(model_name="TransformerConv3D", cam_method="gradcam++"):
    num_slices = {
        "axial": 100,
        "coronal": 100,
        "sagittal": 100,
    }
    original_shapes = {
        "axial": (229, 193),
        "coronal": (193, 193),
        "sagittal": (193, 229)
    }
    entire_dataset_cams = {
        "axial": np.empty((0, num_slices["axial"], 224, 224)),
        "coronal": np.empty((0, num_slices["coronal"], 224, 224)),
        "sagittal": np.empty((0, num_slices["sagittal"], 224, 224))
    }
    # Load the template
    template = nib.load("template/mni_icbm152_t1_tal_nlin_sym_09c.nii")
    template_data = template.get_fdata()
    print(f"Template loaded with shape:{template_data.shape}")
    for fold_num in range(1, 6):
        for plane in ["axial", "coronal", "sagittal"]:
            cams = np.load(f"explainability/{model_name}/fold_{fold_num}/{plane}/{cam_method}.npy")
            print(f"Fold {fold_num} - {plane} - {cams.shape} loaded")
            mean_cams = np.mean(cams, axis=0).reshape((1, num_slices[plane], 224, 224))
            entire_dataset_cams[plane] = np.concatenate((entire_dataset_cams[plane], mean_cams), axis=0)
    print(f"Entire dataset cams shape: {entire_dataset_cams['axial'].shape}")
    entire_dataset_cams = {
        "axial": np.mean(entire_dataset_cams["axial"], axis=0),
        "coronal": np.mean(entire_dataset_cams["coronal"], axis=0),
        "sagittal": np.mean(entire_dataset_cams["sagittal"], axis=0)
    }
    visualization_cams = {
        "axial": [],
        "coronal": [],
        "sagittal": []
    }
    resized_grayscale_cams = {
        "axial": [],
        "coronal": [],
        "sagittal": []
    }
    for plane in entire_dataset_cams:
        for i, cam in enumerate(entire_dataset_cams[plane]):
            cam_rot = np.rot90(entire_dataset_cams[plane][i, :, :], k=1, axes=(1, 0))
            resized_grayscale_cams[plane].append(cv2.resize(cam_rot, original_shapes[plane]))
        for i, cam in enumerate(resized_grayscale_cams[plane]):
            # Get the centered N slices for each plane
            if plane == "sagittal":
                template_slice = template_data[template_data.shape[0] // 2 - num_slices[plane] // 2 + i, :, :]
            elif plane == "coronal":
                template_slice = template_data[:, template_data.shape[1] // 2 - num_slices[plane] // 2 + i, :]
            else:
                template_slice = template_data[:, :, template_data.shape[2] // 2 - num_slices[plane] // 2 + i]
            # Normalize the image between 0 and 1
            template_slice = template_slice - template_slice.min()
            template_slice = template_slice / template_slice.max()
            template_slice = np.float32(template_slice)
            # Convert to RGB
            template_slice = cv2.cvtColor(template_slice, cv2.COLOR_GRAY2RGB)
            cam_result = show_cam_on_image(template_slice, resized_grayscale_cams[plane][i], use_rgb=True,
                                           colormap=cv2.COLORMAP_JET)
            visualization_cams[plane].append(cam_result)
        fold_num = "entire_dataset"
        os.makedirs(os.path.join("explainability", f"{model_name}", f"fold_{fold_num}", plane), exist_ok=True)
    # Save the cam result
    plot_cams(visualization_cams=visualization_cams, model_name=model_name, fold_num=fold_num, cam_method=cam_method)
    cam_3d_map = generate_3d_cam(resized_grayscale_cams, template_data.shape, num_slices)
    # Save the 3D cam
    np.save(f"explainability/{model_name}/fold_{fold_num}/3D_CAM_{cam_method}.npy", cam_3d_map)
    # Generate the explainable MRI
    explainable_mri = generate_explainable_mri(cam_3d_map, template)
    # Save the explainable MRI
    nib.save(explainable_mri, f"explainability/{model_name}/fold_{fold_num}/explainable_mri_{cam_method}.nii")


def compute_xai_metrics(model_name="TransformerConv3D", fold_num="entire_dataset", cam_method="gradcam++",
                        percentile=99.9):
    try:
        cam_map = np.load(f"explainability/{model_name}/fold_{fold_num}/3D_CAM_{cam_method}.npy")
    except:
        print("3D cam map not found")
        return
    print(f"3D cam Map loaded. Shape: {cam_map.shape}")
    try:
        # Load the atlas
        atlas = nib.load("template/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii")
        atlas_data = atlas.get_fdata()
    except:
        print("Atlas not found")
        return
    print(f"Atlas loaded. Shape: {atlas.shape}")
    # Compute the binarized cam map
    binary_map = cam_map > np.percentile(cam_map, percentile)
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
            # Calculate mean, standard deviation, maximum, and minimum values in the cam map for the overlapped region
            overlap_mean = np.mean(cam_map[overlap])
            overlap_std = np.std(cam_map[overlap])
            overlap_max = np.max(cam_map[overlap])
            overlap_min = np.min(cam_map[overlap])
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
    df = pd.DataFrame(data, columns=["Brain Area", "Volume", "Cam Mean", "Cam STD", "Cam Max", "Cam Min",
                                     "Percentage of region"])
    df.sort_values(by="Volume", ascending=False, inplace=True)
    print("Dataframe created\n", df)
    df.to_csv(f"explainability/{model_name}/fold_{fold_num}/xai_metrics_{cam_method}.csv", index=False)
    print("XAI metrics saved")
