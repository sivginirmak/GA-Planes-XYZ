import argparse

def get_parser():
    ''' parse arguments '''

    p = argparse.ArgumentParser()

    ########## required command line inputs ##########
    p.add_argument('--exp_name', type=str, required=True,
                   help='directory to save all output, checkpoints, etc')
    p.add_argument('--supervision', type=str, required=True, choices=['proj', 'sc'],
                   help='use SC labels or 2D projections')
    p.add_argument('--model_type', type=str, required=True, choices=['nerf', 'triplane', "GAplanes"])
    p.add_argument('--convex', type=int, required=True,
                    help='convex model indicator for nerf and triplane / ga planes')
    p.add_argument('--semi_convex', type=int, required=True,
                    help='semi-convex model indicator for triplane / ga planes')


    
    # resolutions and feature dims
    p.add_argument('--Cl', type=int, default=32)
    p.add_argument('--Cp', type=int, default=32)
    p.add_argument('--Cv', type=int, default=8)
    p.add_argument('--Nl', type=int, default=128)
    p.add_argument('--Np', type=int, default=128)
    p.add_argument('--Nv', type=int, default=64)



    ########## general training ##########

    p.add_argument('--exp_type', type=str, default="train")

    p.add_argument('--n_epochs', type=int, default=None,
                   help='number of training epochs')
    p.add_argument('--pe', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0, 
                   help='gpu id to use for training')
    p.add_argument('--lr', type=float, default=5e-4, 
                   help='learning rate')

    p.add_argument('--chunksize', type=int, default=2**12, 
                   help='chunk size')
    p.add_argument('--display_rate', type=int, default=250, 
                   help='display rate')
    p.add_argument('--save_on', type=int, default=1, 
                   help='save models?')
    p.add_argument('--seed', type=int, default=0, 
                   help='random seed')
    
    
    ## add optional comments on the experiment
    p.add_argument("--comments", type=str, default=None)

    args = p.parse_args()
    
    # constants (change)
    args.data_dir = "/home/irmak/convex_nerf/tiny_nerf_data.npz"
    args.sc_vol_dir = "/home/irmak/convex_nerf/sc_vol_N_256.pt"
    args.sc_gt_dir = "/home/irmak/convex_nerf/ground_truth_SC.npy"

    args.savedir = f"Results/{args.exp_name}/"


    # convert ints to bools
    args.convex = bool(args.convex)
    args.semi_convex = bool(args.semi_convex)

    args.pe = bool(args.pe)
    args.save_on = bool(args.save_on)

    
    return args

args = get_parser()
print(args)





