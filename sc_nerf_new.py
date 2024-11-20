import os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import trange
import helpers


#### 3D supervision with space carving

def map_pose_to_proj(ray_pts, ref_lbls, D):

  N = ref_lbls.shape[0]

  xcoord = ray_pts[:, 0]
  ycoord = ray_pts[:, 1]
  zcoord = ray_pts[:, 2]

  k = torch.round((xcoord + D/2) * N/D).to(torch.int)
  l = torch.round((ycoord + D/2) * N/D).to(torch.int)
  m = torch.round((zcoord + D/2) * N/D).to(torch.int)

  cond = lambda x: torch.logical_and(x >= 0, x < N)

  cond_all = torch.logical_and(cond(k), torch.logical_and(cond(l), cond(m)))
  valid_all = torch.where(cond_all)


  k_valid = k[valid_all]
  l_valid = l[valid_all]
  m_valid = m[valid_all]

  proj = ref_lbls[k_valid,l_valid,m_valid]
  return proj



def nerf_forward(ref_volume, carved_labels, nerf_model, encoding_fn, args):
  N = carved_labels.shape[0]
  batches, batched_labels = helpers.prepare_chunks_sc(
    ref_volume.reshape([-1, 3]), carved_labels.flatten(), encoding_function=encoding_fn, chunksize=args.chunksize)
  loss = 0.
  preds = []
  for i, batch in enumerate(batches):
    # print(batch.shape)
    # torch.Size([4096, 63])
    if not(args.model_type == "nerf"):
      batch = batch.unsqueeze(0) #1,4096,3
    lbl = batched_labels[i]
    pred = nerf_model(batch) 

    if (args.model_type == "triplane" and not(args.convex)) or (args.model_type == "GAplanes" and not(args.convex)) :
      #print(pred.shape) #1,4096,1
      pred = pred.squeeze(0)
    preds.append(pred)

    loss += torch.nn.functional.mse_loss(pred.squeeze(), lbl)

  sc_loss = loss / len(batches)
  preds = torch.cat(preds, dim=0).unsqueeze(-1)

  preds = preds.reshape([N,N,N])
  return (sc_loss, preds)


def render_nerf(nerf_labels, height, width, focal, testpose, near, far, device, D, kwargs_sample_stratified):

    rays_o, rays_d = helpers.get_rays(height, width, focal, testpose) # H, W, 3

    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3]) # H*W, 3

    query_points, z_vals = helpers.sample_stratified(rays_o, rays_d, near, far, **kwargs_sample_stratified)

    recon = torch.zeros(height*width).to(device)
    for pix in range(query_points.shape[0]):
      ray = query_points[pix, :, :]
      proj = map_pose_to_proj(ray, nerf_labels, D)
      recon[pix] = torch.sum(proj)

    return recon.reshape([height, width])



def get_psnr(x, gt):
  x = x/torch.max(x)
  gt = gt/torch.max(gt)
  out = 10 * torch.log10(1 / ((x - gt)**2).mean())
  return out

def train(model, encode, optimizer, args, kwargs_sample_stratified, near, far, images, height, width, poses, focal, testpose, testimg, carved_labels, D, ref_coords):


  train_losses_3d = []
  val_losses = []
  val_psnrs = []
  iternums = []

  for i in trange(args.n_epochs):
    model.train()


    (sc_loss, preds) = nerf_forward(ref_coords, carved_labels, model, encode, args)


    sc_loss.backward() 
    optimizer.step()
    optimizer.zero_grad()
    
    train_losses_3d.append(sc_loss.item())

    # print(sc_loss)

    # Evaluate testimg at given display rate.
    if ((i) % args.display_rate == 0) or (i == args.n_epochs -1):
      model.eval()

      rays_o, rays_d = helpers.get_rays(height, width, focal, testpose)

      (sc_loss, preds) = nerf_forward(ref_coords, carved_labels, model, encode, args)
      recon = render_nerf(preds, height, width, focal, testpose, near, far, args.device, D, kwargs_sample_stratified)

      
      imgLoss = torch.nn.functional.mse_loss(recon, testimg)
      print("3D SC Loss:", sc_loss.item())
      print("2D Image Loss:", imgLoss.item())
      val_losses.append(imgLoss.item())
      val_psnr = get_psnr(recon.flatten(), testimg.flatten())
      val_psnrs.append(val_psnr.item())
      iternums.append(i)

      # Plot example outputs
      fig, ax = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 1, 1]})
      ax[0].imshow(recon.detach().cpu().numpy())
      ax[0].set_title(f'Iteration: {i}')
      ax[1].imshow(testimg.reshape([height, width]).detach().cpu().numpy())
      ax[1].set_title(f'SC GT')
      ax[2].plot(iternums, val_psnrs)
      ax[2].set_title('PSNR (val)')
      os.makedirs(args.savedir, exist_ok=True)
      plt.savefig(args.savedir + "val_outs_iter_{}.png".format(i))
      plt.close()

      print("PSNR")
      print(val_psnrs[-1], flush=True)
      if args.save_on and (i == args.n_epochs-1):
        np.save(args.savedir + "val_psnrs.npy", val_psnrs)
        np.save(args.savedir + "val_losses.npy", val_losses)
        np.save(args.savedir + "train_losses_3d.npy", train_losses_3d)
        torch.save(model.state_dict(), args.savedir + f'model_{i}.pt')

  return val_losses, val_psnrs


def threshold(x):
  x = x/torch.max(x)
  m = x.mean()/4
  x[x<m] = 0.
  x[x>=m] = 1.
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

def eval(model, encode, args, kwargs_sample_stratified, near, far, images, height, width, poses, focal, testpose, testimg, carved_labels, D, ref_coords, eval_poses):


  device = args.device
  model.eval()
  iou_scores = []
  psnrs = []

  os.makedirs(args.savedir, exist_ok=True)

  for i,testpose in enumerate(eval_poses):
    testpose = eval_poses[i]

    (sc_loss, preds) = nerf_forward(ref_coords, carved_labels, model, encode, args)
    recon = render_nerf(preds, height, width, focal, testpose, near, far, args.device, D, kwargs_sample_stratified)
    gt_img = render_nerf(carved_labels, height, width, focal, testpose, near, far, args.device, D, kwargs_sample_stratified)


    psnr = get_psnr(recon.flatten(), gt_img.flatten())
    psnrs.append(psnr.item())

    gt_thr = threshold(gt_img)
    recon_thr = threshold(recon)

    gt_thr = gt_thr.reshape([height, width]).detach().cpu().numpy()
    recon_thr = recon_thr.reshape([height, width]).detach().cpu().numpy()

    thr_iou = calculate_iou(recon_thr, gt_thr)
    iou_scores.append(thr_iou.item())

    # Plot example outputs
    fig, ax = plt.subplots(1, 4, figsize=(16,4), gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    ax[0].imshow(recon.detach().cpu().numpy())
    ax[0].set_title(f'Image-recon')
    ax[1].imshow(gt_img.reshape([height, width]).detach().cpu().numpy())
    ax[1].set_title(f'SC GT')
    ax[2].imshow(recon_thr)
    ax[2].set_title(f'recon thr')
    ax[3].imshow(gt_thr)
    ax[3].set_title(f'SC GT thr')
    os.makedirs(args.savedir, exist_ok=True)
    plt.savefig(args.savedir + "evals_img_{}.png".format(i))
    plt.close()

  
  print(psnrs)
  print(iou_scores)

  with open(args.savedir + "eval_results.txt", "w") as file:
    psnr_res = f"mean psnr: {np.mean(psnrs)}, std psnr: {np.std(psnrs)}"
    iou_res = f"mean iou: {np.mean(iou_scores)}, std iou: {np.std(iou_scores)}"
    file.write(psnr_res + "\n")
    file.write(iou_res)


  return psnrs, iou_scores


