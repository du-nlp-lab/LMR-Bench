import argparse
import os
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

import lib
import monkey_patch as mp 
import pickle

# print('torch', version('torch'))
# print('transformers', version('transformers'))
# print('accelerate', version('accelerate'))
# print('# of gpus: ', torch.cuda.device_count())

def run_exp1(layer_id, layers, model, val_transform, savedir, model_family, model_size):
    """
    Run 3D feature visualization for a specific layer in a Vision Transformer (ViT) model.

    This function extracts the absolute feature activations from a specified transformer layer,
    visualizes them in 3D, and saves both the input parameters and feature activations as 
    serialized pickle files for reproducibility and analysis.

    Args:
        layer_id (int): The 1-based index of the transformer layer to visualize. 
                        The function internally converts this to a 0-based index.
        layers (list): A list of transformer layers from the loaded ViT model.
        model (torch.nn.Module): The full ViT model for which the features are being visualized.
        val_transform (callable): A transformation function (e.g., torchvision.transforms) 
                                  to preprocess the input image before feeding it to the model.
        savedir (str): Directory path where the input parameters and feature activations 
                       will be saved as pickle files.
        model_family (str): A string identifier for the model family (e.g., 'dinov2', 'vit').
        model_size (str): A string identifier for the model size (e.g., 'large', 'base').

    Returns:
        None

    Side Effects:
        - Saves a pickle file containing the input parameters as 'run_exp1_inputs_layer_{layer_id}.pkl'
        - Saves a pickle file containing the extracted feature activations as 'feat_abs_layer_{layer_id}.pkl'
        - Generates a 3D visualization of the extracted feature activations and saves it to the specified directory."""


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_family', type=str)
#     parser.add_argument('--model_size', type=str)
#     parser.add_argument("--layer_id", type=int, default=1)
#     parser.add_argument('--exp1', action="store_true", help="plot 3d feature")
#     parser.add_argument('--exp2', action="store_true", help="layerwise analysis")
#     parser.add_argument('--exp3', action="store_true", help="test original and fix-reg-mean accuracy")
#     parser.add_argument('--imagenet_dir', type=str, default="/home/mingjies/imagenet-data/val")
#     parser.add_argument('--linear_head_path', type=str, default="/data/locus/project_data/project_data2/mingjies/dinov2")
#     parser.add_argument('--reg_feat_mean', type=str, default="assets/reg_feat_mean/")
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--num_imgs_mean', type=int, default=10)
#     parser.add_argument('--savedir', type=str)
#     args = parser.parse_args()

#     torch.manual_seed(args.seed)

#     if not os.path.exists(args.savedir):
#         os.makedirs(args.savedir)

#     model, layers, val_transform = lib.load_vit(args)

#     if args.exp1:
#         run_exp1(args.layer_id, layers, model, val_transform, args.savedir, args.model_family, args.model_size)
#         # layer_id = args.layer_id - 1
#         # layer = layers[layer_id]
#         # mp.enable_vit_custom_block(layer, layer_id)

#         # img_path = os.path.join("assets", f"bird.png")
#         # img = Image.open(img_path)
#         # img = val_transform(img).unsqueeze(0).cuda()

#         # with torch.no_grad():
#         #     output = model(img)

#         # feat_abs = layers[layer_id].feat.abs()

#         # lib.plot_3d_feat_vit(feat_abs, layer_id, args.model_family, args.model_size, args.savedir)
#         # torch.save(stats, os.path.join(args.savedir, f"stats.pt"))

#     elif args.exp2:
#         for layer_id in range(len(layers)):
#             layer = layers[layer_id]
#             mp.enable_vit_custom_block(layer, layer_id)

#         dataset = datasets.ImageFolder(args.imagenet_dir, transform=val_transform)

#         stats = []
#         for img_idx in range(args.num_imgs_mean):
#             print("img_idx", img_idx)
#             images, target = dataset[img_idx]
#             images = images.unsqueeze(0).cuda()

#             with torch.no_grad():
#                 output = model(images)

#             layer_stats_np = np.zeros((4, len(layers)))
#             for layer_id in range(len(layers)):
#                 feat_abs = layers[layer_id].feat.abs()
#                 sort_res = torch.sort(feat_abs.flatten(), descending=True)
#                 layer_stats_np[:3, layer_id] = sort_res.values[:3]
#                 layer_stats_np[3, layer_id] = torch.median(feat_abs)

#             stats.append(layer_stats_np)

#         lib.plot_layer_ax_vit(np.mean(stats, axis=0), args.model_family, args.model_size, args.savedir)

#     elif args.exp3:
#         linear_head = lib.load_dinov2_linear_head(args)
#         lib.setup_dinov2_model_for_eval(model, linear_head)

#         dataset = datasets.ImageFolder(args.imagenet_dir, transform=val_transform)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, num_workers=8, pin_memory=False)

#         f = open(os.path.join(args.savedir, "eval.txt"), "w")
#         top1_acc = lib.test_imagenet(model, dataloader)
#         print(f"{args.model_family} ViT-{args.model_size} original accuracy: {top1_acc}", file=f, flush=True)

#         lib.fix_reg_mean(args, model)
#         top1_acc = lib.test_imagenet(model, dataloader)
#         print(f"{args.model_family} ViT-{args.model_size} fix-reg-mean accuracy: {top1_acc}", file=f, flush=True)