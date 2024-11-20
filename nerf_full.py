import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from tqdm import trange
import torch.nn.functional as F

import helpers

from nerf_models import load_kmeans_svm_convexNet, MyFCConvexNet, Nerf



def nerf_forward(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  target_img: torch.Tensor,
  nerf_model: nn.Module,
  encoding_fn: Callable[[torch.Tensor], torch.Tensor],
  kwargs_sample_stratified: dict = None,
  chunksize: int = 2**15
) -> Tuple[torch.Tensor, torch.Tensor]:
  r"""
  Compute forward pass through model(s).
  """

  # Set no kwargs if none are given.
  if kwargs_sample_stratified is None:
    kwargs_sample_stratified = {}


  # Sample query points along each ray.
  query_points, z_vals = helpers.sample_stratified(
      rays_o, rays_d, near, far, **kwargs_sample_stratified)
  
  # Prepare batches.
  batches = helpers.prepare_chunks(query_points, encoding_function=encoding_fn,chunksize=chunksize)
  
  predictions = []


  for batch in batches:

    pred = nerf_model(batch)
    predictions.append(pred)

  raw = torch.cat(predictions, dim=0).unsqueeze(-1)
  raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]]).squeeze(-1)
  return raw 





def init_models():
  r"""
  Initialize models, encoders, and optimizer for NeRF training.
  """
  # Models
  # in_features, hid_features, out_features, nonlin
  if useEncoder == True:

    encoder = helpers.PositionalEncoder(d_input, n_freqs, log_space=True)
    encode = lambda x: encoder(x)
    in_features = encoder.d_output

  else:
    encode = None
    in_features = 3
  
  if useConvexModel:

    if useSCplanes:
      model = load_kmeans_svm_convexNet(device=device, pe=useEncoder)

    else:
      model = MyFCConvexNet(in_features=in_features, hid_features=512, nl="relu")
  else:
    model = Nerf(in_features=in_features, hid_features=256, out_features=1)
  
  model.to(device)
  model_params = list(model.parameters())

  optimizer = torch.optim.AdamW(model_params, lr=lr, betas=(0.8, 0.999)) #lr = 0.001?


  return model, optimizer, encode



def render_image(raw):
  # raw.shape = #([10000, 512]) (H*W, num_samples_along_each_ray)
  image = torch.mean(raw, dim=-1)
  return image

def train(images, testimg):
  r"""
  Launch training session for NeRF.
  """

  train_psnrs = []
  train_losses = []
  val_psnrs = []
  val_losses = []
  iternums = []

  for i in trange(n_iters):
    model.train()


    # Randomly pick an image as the target.
    target_img_idx = np.random.randint(images.shape[0])
    target_img = images[target_img_idx].to(device)

    height, width = target_img.shape[:2]
    target_pose = poses[target_img_idx].to(device)
    rays_o, rays_d = helpers.get_rays(height, width, focal, target_pose)

    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3]) # H*W, 3
    
    # print(target_img.shape) #torch.Size([100, 100])
    target_img = target_img.reshape([-1]) # H*W
    # print(target_img.shape) #torch.Size([10000])

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    raw = nerf_forward(rays_o, rays_d,
                           near, far, target_img, model, encode,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           chunksize=chunksize)

    # Check for any numerical issues.
    
    if torch.isnan(raw).any():
        print(f"! [Numerical Alert] predictions contain NaN.")
    if torch.isinf(raw).any():
        print(f"! [Numerical Alert] predictions contain Inf.")

    pred_img = render_image(raw)
    imgLoss = torch.nn.functional.mse_loss(pred_img, target_img)
    train_losses.append(imgLoss.item())
    imgLoss.backward() #trying the other loss now
    optimizer.step()
    optimizer.zero_grad()
    
    psnr = -10. * torch.log10(imgLoss)
    train_psnrs.append(psnr.item())

    # Evaluate testimg at given display rate.
    if ((i) % display_rate == 0) or (i == n_iters-1):
      model.eval()

      rays_o, rays_d = helpers.get_rays(height, width, focal, testpose)

      
      rays_o = rays_o.reshape([-1, 3])
      rays_d = rays_d.reshape([-1, 3])
      testimg = testimg.reshape([-1])

      raw = nerf_forward(rays_o, rays_d,
                           near, far, testimg, model, encode,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           chunksize=chunksize)
      

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
      ax[2].set_title('PSNR (train=red, val=blue')

      
      os.makedirs(savedir, exist_ok=True)
      plt.savefig(savedir + "val_outs_iter_{}.png".format(i))
      plt.close()

      print("PSNR")
      print(val_psnrs[-1], flush=True)
      np.save(savedir + "val_psnrs.npy", val_psnrs)
      np.save(savedir + "train_psnrs.npy", train_psnrs)
      np.save(savedir + "val_losses.npy", val_losses)
      np.save(savedir + "train_losses.npy", train_losses)


  return True, train_psnrs, val_psnrs

def main(args):

  savedir = args.savedir   #"/home/irmak/convex_nerf_submit/svm_convexNerf_pe/"

  # For repeatability
  seed = 0
  torch.manual_seed(seed)
  np.random.seed(seed)
  # print(torch.__version__)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print("running on device", device)
  data = np.load('/home/irmak/convex_nerf/tiny_nerf_data.npz')


  near, far = 2., 6.

  n_training = 100
  testimg_idx = 101



  # Gather as torch tensors
  images = torch.from_numpy(data['images'][:n_training]).to(device)
  height, width = images.shape[1:3]
  images = torch.norm(images, dim=-1)
  images[images!=0] = 1 # convert to 1-0 masks
  print(images.shape) #torch.Size([100, 100, 100])

  poses = torch.from_numpy(data['poses']).to(device)
  focal = torch.from_numpy(data['focal']).to(device)
  testimg = torch.from_numpy(data['images'][testimg_idx]).to(device)
  testpose = torch.from_numpy(data['poses'][testimg_idx]).to(device)

  testimg = torch.norm(testimg, dim=-1)
  testimg[testimg!=0] = 1 # convert to 1-0 masks



  d_input = 3           # Number of input dimensions
  n_freqs = 10 

  # Stratified sampling
  n_samples = 64*8        # Number of spatial samples per ray
  #n_samples = 64*4
  perturb = False         # If set, applies noise to sample positions
  inverse_depth = False  # If set, samples points linearly in inverse depth

  useEncoder = True
  useConvexModel = True
  useSCplanes = True

  # Optimizer
  lr = 5e-4  # Learning rate

  # Training
  n_iters = 10000
  batch_size = 2**14          # Number of rays per gradient step (power of 2)
  chunksize = 2**12           # Modify as needed to fit in GPU memory

  display_rate = 250          # Display test output every X epochs



  # We bundle the kwargs for various functions to pass all at once.
  kwargs_sample_stratified = {
      'n_samples': n_samples,
      'perturb': perturb,
      'inverse_depth': inverse_depth
  }



  model, optimizer, encode = init_models()

  success, train_psnrs, val_psnrs = train(images, testimg)


  print('')
  print(f'Done!')

  print("VAL PSNR")
  print(val_psnrs[-1])

  torch.save(model.state_dict(), savedir + 'model.pt')

