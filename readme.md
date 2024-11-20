<!-- model types:
- (semi) convex triplane
- convex ga planes (both fully convex and semi convex)
- nerf (baseline)
- convex nerf (works only with SC supervision, gates as k-mans svm weights) 

triplane_models:

ConvexTriplane(addlines=False)-- fully convex triplane
ConvexTriplane(addlines=True)-- fully convex GA plane

MiniTriplane(addlines=False, convex=False) -- baseline triplane (we modified it)
MiniTriplane(addlines=False, convex=True) -- semi convex triplane 
MiniTriplane(addlines=True, convex=False) -- GA plane 
MiniTriplane(addlines=True, convex=True) -- semi convex GA plane 

For projection supervision, run: triplane_new.py with the desired model
Foe SC supervision, run: sc_nerf_new.py
It has the following options:

if useTriplane:
model = MiniTriplane(convex=True, addlines=True) #addlines=True
elif useConvexTriplane:
model = ConvexTriplane(addlines=True)
elif useConvex:
if useNonconvexWeights:
    model = loadModel.get_model(device)
    #/home/irmak/convex_nerf/sc_nerf_4/model.pt
elif useSCplanes:
    model = load_SC_convex_net(device=device, hid_dim=256, inp_dim=in_features, num_reps=4)
else:
    model = MyFCConvexNet(in_features=in_features, hid_features=1024)
# model = loadModel.MyConvexNetFCThreeLayer(input_dim=in_features)
else:
model = Nerf(in_features=in_features, hid_features=256, out_features=1) -->

<!-- Project repo for 3D object segmentation with convex MLPs -->

Project repo for 3D object segmentation with convex Nerfs, Triplanes, GAplanes. K-planes and TensorF implemented for comparison
