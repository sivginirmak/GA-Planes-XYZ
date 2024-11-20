import torch
from torch import nn
import numpy as np


class ConvexNet(nn.Module):
    def __init__(self, in_features, hid_features):
        super().__init__()
        self.filters = hid_features
        self.input_size = in_features

        self.fc1 = nn.Linear(self.input_size, self.filters, bias=True)
        torch.manual_seed(100)
        self.fc2 = nn.Linear(self.input_size, self.filters, bias=True)

        self.fc2.weight.requires_grad = False

    def forward(self, x):

        x = self.fc1(x)*(self.fc2(x)>=0)
        x = torch.sum(x[:,:x.shape[1]],dim=-1)
        return x

class MiniTriplane(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, addlines=False, convex=False, Cp=32, Cl=32, Cv=6, Nv=64, Nl=128, Np=128, cat=True):
        super().__init__()

        self.addlines = addlines
        self.cat = cat
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, Cp, Np, Np)*0.001) for _ in range(3)])

        if addlines:
            self.lines = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, Nl, 1)*0.001) for _ in range(3)])
            self.volume = nn.Parameter(torch.randn(1, Cv, Nv, Nv, Nv)*0.001)

        if addlines:
            if cat:
                in_dim = 3*Cl + 3*Cp + Cv
            else:
                in_dim = Cl+Cv+Cp
        else:
            in_dim = Cp

        if not(convex):
            self.net = nn.Sequential(                
                nn.Linear(in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, output_dim),
            )
        else:
            self.net = ConvexNet(in_features=in_dim, hid_features=128)


    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def sample_volume(self, coords3d, plane):
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords3d.reshape(coords3d.shape[0], 1, 1, -1, coords3d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_features = sampled_features.squeeze(2)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])

        features = xy_embed + yz_embed + xz_embed

        if self.addlines:

            xs = torch.stack([coordinates[..., 0],(coordinates[..., 0])],dim=-1)
            ys = torch.stack([coordinates[..., 1],(coordinates[..., 1])],dim=-1)
            zs = torch.stack([coordinates[..., 2],(coordinates[..., 2])],dim=-1)

            x_embed = self.sample_plane(xs, self.lines[0])
            y_embed = self.sample_plane(ys, self.lines[1])
            z_embed = self.sample_plane(zs, self.lines[2])

            xyz_embed = self.sample_volume(coordinates, self.volume)
            if self.cat:
                features = torch.cat([x_embed, y_embed, z_embed, xy_embed, yz_embed, xz_embed, xyz_embed], dim=-1)
            else:
                line_features = x_embed + y_embed + z_embed
                features = torch.cat([features, line_features, xyz_embed], dim=-1)
         
        return self.net(features)


    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l
    
    

class ConvexTriplane(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, Cl=64, Cp=32, Cv=16, Nl=128, Np=128, Nv=64,addlines=False):
        super().__init__()

        self.addlines = addlines
        self.planes = nn.ParameterList([nn.Parameter(torch.randn(1, Cp, Np, Np)*0.001) for _ in range(3)])
        

        if addlines:
            self.lines = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, Nl, 1)*0.001) for _ in range(3)])
            self.volume = nn.Parameter(torch.randn(1, Cv, Nv, Nv, Nv)*0.001)
            torch.manual_seed(100)
            self.lines_g = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, Nl, 1)*0.001, requires_grad=False) for _ in range(3)])
            self.volume_g = nn.Parameter(torch.randn(1, Cv, Nv, Nv, Nv)*0.001, requires_grad=False)
        
        torch.manual_seed(100)
        self.planes_g = nn.ParameterList([nn.Parameter(torch.randn(1, Cp, Np, Np)*0.001, requires_grad=False) for _ in range(3)])


    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def sample_volume(self, coords3d, plane):
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords3d.reshape(coords3d.shape[0], 1, 1, -1, coords3d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_features = sampled_features.squeeze(2)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.planes[0]).squeeze(0)
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.planes[1]).squeeze(0)
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.planes[2]).squeeze(0)

        xy_gate = self.sample_plane(coordinates[..., 0:2], self.planes_g[0]).squeeze(0) # B, C
        yz_gate = self.sample_plane(coordinates[..., 1:3], self.planes_g[1]).squeeze(0)
        xz_gate = self.sample_plane(coordinates[..., :3:2], self.planes_g[2]).squeeze(0)

        features = torch.sum(xy_embed * (xy_gate >= 0)+ yz_embed * (yz_gate >= 0) + xz_embed * (xz_gate >= 0), dim=-1)

        # print(features.shape) # torch.Size([4096])
        if self.addlines:

            xs = torch.stack([coordinates[..., 0],torch.zeros_like(coordinates[..., 0])],dim=-1)
            ys = torch.stack([coordinates[..., 1],torch.zeros_like(coordinates[..., 1])],dim=-1)
            zs = torch.stack([coordinates[..., 2],torch.zeros_like(coordinates[..., 2])],dim=-1)
            # print(xs.shape) torch.Size([1, 1024, 2])

            x_embed = self.sample_plane(xs, self.lines[0]).squeeze(0)
            y_embed = self.sample_plane(ys, self.lines[1]).squeeze(0)
            z_embed = self.sample_plane(zs, self.lines[2]).squeeze(0)

            x_gate = self.sample_plane(xs, self.lines_g[0]).squeeze(0)
            y_gate = self.sample_plane(ys, self.lines_g[1]).squeeze(0)
            z_gate = self.sample_plane(zs, self.lines_g[2]).squeeze(0)

            xyz_embed = self.sample_volume(coordinates, self.volume).squeeze(0)
            xyz_gate = self.sample_volume(coordinates, self.volume_g).squeeze(0)

            line_features = torch.sum(x_embed * (x_gate >= 0) + y_embed * (y_gate >= 0) + z_embed * (z_gate >= 0), dim=-1) 
            volume_features = torch.sum(xyz_embed * (xyz_gate >= 0), dim=-1)
            features = features + line_features + volume_features
         
        return features





class Kplanes(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, linear=False, Cp=32, Np=128):
        super().__init__()

        self.range = 0.4
        self.shift = 0.1
        self.embeddings = nn.ParameterList([nn.Parameter(torch.rand(1, Cp, Np, Np)*self.range + self.shift) for _ in range(3)])
        in_dim = Cp 
        if not(linear):
            self.net = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, output_dim)
            )
        else:
            self.net = nn.Linear(in_dim, output_dim)


    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features


    def forward(self, coordinates):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])

        features = xy_embed * yz_embed * xz_embed
        return self.net(features)


