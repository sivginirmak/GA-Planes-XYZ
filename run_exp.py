import torch
import numpy as np
from nerf_models import *
from triplane_models import *

import parser
import helpers
import sc_nerf_new
import nerf_fcns



# init models (merged!)
def init_models(args):
  # Initialize models, encoders, and optimizer for NeRF training.
  device = args.device
  if args.pe:
    d_input = 3           # Number of input dimensions
    n_freqs = 10
    encoder = helpers.PositionalEncoder(d_input, n_freqs, log_space=True)
    encode = lambda x: encoder(x)
    in_features = encoder.d_output

  else:
    encode = None
    in_features = 3
  
  # proj / sc model selection
  if args.model_type == "nerf":
    ## uses PE instead of encoding through feature grids!
    model = Nerf(in_features=in_features, hid_features=256, out_features=1)
 
  elif args.model_type == "triplane":
    if args.semi_convex:
      ## convex MLP decoder
      model = MiniTriplane(convex=True, addlines=False, Cp=args.Cp, Np=args.Np) # Cp=C
    elif args.convex: 
      ## fused grids & decoder weights
      model = ConvexTriplane(addlines=False, Cp=args.Cp, Np=args.Np)
    else:
      ## nonconvex MLP decoder
      model = MiniTriplane(convex=False, addlines=False, Cp=args.Cp, Np=args.Np)

  elif args.model_type == "GAplanes":
    if args.semi_convex:
      model = MiniTriplane(convex=True, addlines=True, Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv) # Cl=Cl, Cp=Cp, Cv=Cv, Nv=Nv
    elif args.convex:
      model = ConvexTriplane(addlines=True, Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv)
    else:
      model = MiniTriplane(convex=False, addlines=True, Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv)

  print(model)
  model.to(device)

  return model, encode








def main():
    # run any type of training with this master main code!!

    args = parser.get_parser()

    # set cuda device, random seed
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(device)
    args.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ## needed for all nerf / sc / triplane etc.
    configs = {} # bundle all needed variables for training etc.


    data = np.load(args.data_dir) ## load scene data
    near, far = 2., 6.
    n_training = 100
    testimg_idx = 101

    #### convert to segmentations
    # Gather as torch tensors
    images = torch.from_numpy(data['images'][:n_training]).to(device)
    height, width = images.shape[1:3]
    images = torch.norm(images, dim=-1)
    images[images!=0] = 1 # convert to 1-0 masks

    eval_images = torch.from_numpy(data['images'][testimg_idx:]).to(device)
    eval_images = torch.norm(eval_images, dim=-1)
    eval_images[eval_images!=0] = 1 # convert to 1-0 masks

    poses = torch.from_numpy(data['poses']).to(device)
    eval_poses = poses[testimg_idx:]

    focal = torch.from_numpy(data['focal']).to(device)
    testimg = torch.from_numpy(data['images'][testimg_idx]).to(device)
    testpose = torch.from_numpy(data['poses'][testimg_idx]).to(device)

    testimg = torch.norm(testimg, dim=-1)
    testimg[testimg!=0] = 1 # convert to 1-0 masks


    configs["near"] = near
    configs["far"] = far
    configs["images"] = images
    configs["height"] = height
    configs["width"] = width
    configs["poses"] = poses
    configs["focal"] = focal
    configs["testpose"] = testpose
    configs["testimg"] = testimg

    # Stratified sampling
    n_samples = 64*8        # Number of spatial samples per ray
    perturb = False         # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    kwargs_sample_stratified = {
      'n_samples': n_samples,
      'perturb': perturb,
      'inverse_depth': inverse_depth
    }

    if args.supervision == "sc":
        # load additional files  and define ref coords!!
        gt = np.load(args.sc_gt_dir)
        gt = torch.from_numpy(gt).to(device)
        carved_labels = torch.load(args.sc_vol_dir).to(device)
        carved_labels = carved_labels[25:-25, 25:-25, 25:-25]
        ts = poses[:,:3,-1]
        D = torch.mean(torch.norm(ts,dim=-1))
        D = D*0.8

        configs["testimg"] = gt
        configs["carved_labels"] = carved_labels
        configs["D"] = D

        # ref coords
        N = carved_labels.shape[0]
        xr = torch.linspace(-D/2, D/2, N)
        yr = torch.linspace(D/2, -D/2, N)
        zr = torch.linspace(-D/2, D/2, N)
        xx, yy, zz = torch.meshgrid(xr, yr, zr, indexing='ij')

        ref_coords = torch.stack([xx,yy,zz]).permute([1,2,3,0]).to(device) # torch.Size([512, 512, 512, 3])
        configs["ref_coords"] = ref_coords

    if args.exp_type == "train":
      ### model selection (get from parser)
      model, encode = init_models(args)    
      print(helpers.count_parameters(model))

      torch.manual_seed(args.seed)
      ## optimizer
      model_params = list(model.parameters())
      optimizer = torch.optim.AdamW(model_params, lr=args.lr)

      if args.supervision == "sc":
        val_losses, val_psnrs = sc_nerf_new.train(model, encode, optimizer, args, kwargs_sample_stratified, **configs)
      else:
        train_psnrs, val_psnrs = nerf_fcns.train(model, encode, optimizer, args, kwargs_sample_stratified, **configs)
      
      # save the parser parameters
      with open(args.savedir + 'options.txt', 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    elif args.exp_type == "eval_iou":
      model, encode = init_models(args)
      configs["eval_poses"] = eval_poses
      ## load trained & saved model
      parameterDict = torch.load(f"Results/{args.exp_name}/model_{args.n_epochs - 1}.pt")

      model.load_state_dict(parameterDict)
      if args.supervision == "proj":
        configs["eval_images"] = eval_images
        configs["plot_nm"] = "recon_model_iou_all.png"
        thr_iou = nerf_fcns.eval_all(model, encode, args, kwargs_sample_stratified, **configs)

      else:
        psnrs, iou_scores = sc_nerf_new.eval(model, encode, args, kwargs_sample_stratified, **configs)



if __name__ == '__main__':
    main()