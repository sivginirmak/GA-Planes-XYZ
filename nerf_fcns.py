from typing import Optional, Tuple, List, Union, Callable
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from tqdm import trange

import helpers

def nerf_forward(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  target_img: torch.Tensor,
  nerf_model: nn.Module,
  encoding_fn: Callable[[torch.Tensor], torch.Tensor],
  args,
  kwargs_sample_stratified: dict = None
) -> Tuple[torch.Tensor, torch.Tensor]:
  r"""
  Compute forward pass through model(s).
  """

  # Set no kwargs if none are given.
  if kwargs_sample_stratified is None:
    kwargs_sample_stratified = {}

  # Sample query points along each ray.
  query_points, z_vals = helpers.sample_stratified(
      rays_o, rays_d, near, far, **kwargs_sample_stratified)  # torch.Size([10000, 512, 3])

  # Prepare batches.
  batches = helpers.prepare_chunks(query_points, encoding_function=encoding_fn, chunksize=args.chunksize)

  predictions = []

  for batch in batches:

    if not(args.model_type == "nerf"):
        batch = batch.unsqueeze(0)

    pred = nerf_model(batch)

    predictions.append(pred)
  raw = torch.cat(predictions, dim=0).unsqueeze(-1)

  raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]]).squeeze(-1)

  return raw 


def render_image(raw):
  # raw.shape = #([10000, 512]) (H*W, num_samples_along_each_ray)
  image = torch.mean(raw, dim=-1)
  return image


def train(model, encode, optimizer, args, kwargs_sample_stratified, near, far, images, height, width, poses, focal, testpose, testimg):
  r"""
  Launch training session for NeRF.
  """

  train_psnrs = []
  train_losses = []
  val_psnrs = []
  val_losses = []
  iternums = []
  device = args.device
  for i in trange(args.n_epochs):
    model.train()

    
    # Randomly pick an image as the target.
    target_img_idx = np.random.randint(images.shape[0])
    target_img = images[target_img_idx].to(device)
    height, width = target_img.shape[:2]
    target_pose = poses[target_img_idx].to(device)
    rays_o, rays_d = helpers.get_rays(height, width, focal, target_pose)

    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3]) # H*W, 3
    

    target_img = target_img.reshape([-1]) # H*W


    raw = nerf_forward(rays_o, rays_d,
                           near, far, target_img, model, encode, args,
                           kwargs_sample_stratified=kwargs_sample_stratified)

    # Check for any numerical issues.
    
    if torch.isnan(raw).any():
        print(f"! [Numerical Alert] predictions contain NaN.")
    if torch.isinf(raw).any():
        print(f"! [Numerical Alert] predictions contain Inf.")



    pred_img = render_image(raw)
    imgLoss = torch.nn.functional.mse_loss(pred_img, target_img)
    train_losses.append(imgLoss.item())

    imgLoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    psnr = -10. * torch.log10(imgLoss)
    train_psnrs.append(psnr.item())

    # Evaluate testimg at given display rate.
    if ((i) % args.display_rate == 0) or (i == args.n_epochs-1):
      model.eval()

      rays_o, rays_d = helpers.get_rays(height, width, focal, testpose)
      
      rays_o = rays_o.reshape([-1, 3])
      rays_d = rays_d.reshape([-1, 3])
      testimg = testimg.reshape([-1])



      raw = nerf_forward(rays_o, rays_d,
                           near, far, testimg, model, encode, args,
                           kwargs_sample_stratified=kwargs_sample_stratified)
      

      pred_img = render_image(raw)
      imgLoss = torch.nn.functional.mse_loss(pred_img, testimg)
      val_losses.append(imgLoss.item())

      print("Loss:", imgLoss.item())
      val_psnr = -10. * torch.log10(imgLoss)
      val_psnrs.append(val_psnr.item())
      iternums.append(i)

      # Plot example outputs
      fig, ax = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 1, 1]})
      ax[0].imshow(pred_img.reshape([height, width]).detach().cpu().numpy())
      ax[0].set_title(f'Iteration: {i}')
      ax[1].imshow(testimg.reshape([height, width]).detach().cpu().numpy())
      ax[1].set_title(f'Target')
      ax[2].plot(range(0, i + 1), train_psnrs, 'r')
      ax[2].plot(iternums, val_psnrs, 'b')
      ax[2].set_title('PSNR (train=red, val=blue)')

      
      os.makedirs(args.savedir, exist_ok=True)
      plt.savefig(args.savedir + "val_outs_iter_{}.png".format(i))
      plt.close()

      print("PSNR")
      print(val_psnrs[-1], flush=True)
      if args.save_on and (i == args.n_epochs-1):
        np.save(args.savedir + "val_psnrs.npy", val_psnrs)
        np.save(args.savedir + "train_psnrs.npy", train_psnrs)
        np.save(args.savedir + "val_losses.npy", val_losses)
        np.save(args.savedir + "train_losses.npy", train_losses)
        torch.save(model.state_dict(), args.savedir + f'model_{i}.pt')


  return train_psnrs, val_psnrs



def threshold(x):
  x = x/torch.max(x)
  x[x<0.25] = 0.
  x[x>=0.25] = 1.
  return x

def calculate_iou(binary_image1, binary_image2):
  # Calculate the intersection and union
  intersection = np.logical_and(binary_image1, binary_image2).sum()
  union = np.logical_or(binary_image1, binary_image2).sum()
  
  # Avoid division by zero
  if union == 0:
      return 0.0
  
  # Calculate IoU
  iou = intersection / union
  return iou



def eval_all(model, encode, args, kwargs_sample_stratified, near, far, images, height, width, poses, focal, testpose, testimg, plot_nm, eval_images, eval_poses):
  r"""
  Launch training session for NeRF.
  """

  device = args.device
  model.eval()
  iou_scores = []

  for i,testimg in enumerate(eval_images):
    testpose = eval_poses[i]
    rays_o, rays_d = helpers.get_rays(height, width, focal, testpose)
  
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    testimg = testimg.reshape([-1])
    raw = nerf_forward(rays_o, rays_d,
                          near, far, testimg, model, encode, args,
                          kwargs_sample_stratified=kwargs_sample_stratified)
  

    pred_img = render_image(raw)

    thr_pred = threshold(pred_img)
    pred_img = pred_img.reshape([height, width]).detach().cpu().numpy()
    thr_pred = thr_pred.reshape([height, width]).detach().cpu().numpy()
    testimg = testimg.reshape([height, width]).detach().cpu().numpy()

    thr_iou = calculate_iou(thr_pred, testimg)
    iou_scores.append(thr_iou.item())
    if i==0:
      recon = pred_img
      recon_thr = thr_pred
      gt = testimg

  os.makedirs(args.savedir, exist_ok=True)
  print(iou_scores)

  fig, ax = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 1, 1]})
  ax[0].imshow(recon)
  ax[0].set_title(f'Rendering')
  ax[1].imshow(recon_thr)
  ax[1].set_title(f'mean IoU: {np.mean(iou_scores)}, IoU std: {np.std(iou_scores)}')
  ax[2].imshow(gt)
  ax[2].set_title(f'Target')

  plt.savefig(args.savedir + plot_nm)
  plt.close()

  with open(args.savedir + "iou_scores.txt", "w") as file:
    text_data = f'IoU mean: {np.mean(iou_scores)}, IoU std: {np.std(iou_scores)}'
    file.write(text_data)

  return iou_scores

