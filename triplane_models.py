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


class RankOnePlanes(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, convex=False, Cl=32, Cv=6, Nv=64):
        super().__init__()
        
        # self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, 128, 128)*0.01) for _ in range(3)])
        self.range = 0.4
        self.shift = 0.1
        self.lines = nn.ParameterList([nn.Parameter(torch.rand(1, Cl, 128, 1)*self.range + self.shift, requires_grad=True) for _ in range(3)])
        # self.volume = nn.Parameter(torch.randn(1, Cv, Nv, Nv, Nv)*0.001)


        if not(convex):
            self.net = nn.Sequential(
                
                # nn.Linear(Cl, 128), # Cl+Cv+Cp
                # nn.ReLU(inplace=True),
                                
                nn.Linear(Cl, 128),
                nn.ReLU(inplace=True),
                
                nn.Linear(128, output_dim),
                # nn.Linear(Cl, output_dim)
            )
            # self.net2 = nn.Sequential(nn.Linear(Cl, 128), nn.ReLU(inplace=True), nn.Linear(128, output_dim))
            # self.net3 = nn.Sequential(nn.Linear(Cl, 128), nn.ReLU(inplace=True), nn.Linear(128, output_dim))
        else:

            self.net = ConvexNet(in_features=Cl, hid_features=128)
            # self.net2 = ConvexNet(in_features=Cl, hid_features=128)
            # self.net3 = ConvexNet(in_features=Cl, hid_features=128)


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
        
        # xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        # yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        # xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])

        # features = xy_embed + yz_embed + xz_embed
        # print(features.shape) # torch.Size([1, 4096, 32])

        # print(torch.max(coordinates[..., 0]), torch.min(coordinates[..., 0]))
        # print(torch.max(coordinates[..., 1]), torch.min(coordinates[..., 1]))
        # print(torch.max(coordinates[..., 2]), torch.min(coordinates[..., 2]))
        x_coords = coordinates[..., 0]
        y_coords = coordinates[..., 1]
        z_coords = coordinates[..., 2]
        ##### (0,x)
        # xs = torch.stack([torch.zeros_like(coordinates[..., 0]), coordinates[..., 0]],dim=-1)
        # ys = torch.stack([torch.zeros_like(coordinates[..., 1]), coordinates[..., 1]],dim=-1)
        # zs = torch.stack([torch.zeros_like(coordinates[..., 2]), coordinates[..., 2]],dim=-1)
        ###### (x,x)
        # xs = torch.stack([x_coords, x_coords],dim=-1)
        # ys = torch.stack([y_coords, y_coords],dim=-1)
        # zs = torch.stack([z_coords, z_coords],dim=-1)
        ##### (1,x)
        xs = torch.stack([torch.ones_like(x_coords), x_coords],dim=-1)
        ys = torch.stack([torch.ones_like(y_coords), y_coords],dim=-1)
        zs = torch.stack([torch.ones_like(z_coords), z_coords],dim=-1)

        # print(xs.shape) torch.Size([1, 1024, 2])

        x_embed = self.sample_plane(xs, self.lines[0])
        y_embed = self.sample_plane(ys, self.lines[1])
        z_embed = self.sample_plane(zs, self.lines[2])

        # xyz_embed = self.sample_volume(coordinates, self.volume)

        features = (x_embed + y_embed + z_embed)
        features2 = (x_embed * y_embed + x_embed * z_embed + y_embed * z_embed) * 1/self.range
        features3 =  x_embed * y_embed * z_embed * 1/(self.range**2)

        # features += x_embed * yz_embed + y_embed * xz_embed + z_embed * xy_embed
        # features += yz_embed + xz_embed + xy_embed
        # f = features + features2 + features3

        return self.net(features + features2 + features3) #+ self.net2(features2) + self.net3(features3)


class RankOnePlanesNew(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, convex=False, Cl=32, Nl=128):
        super().__init__()
        
        # self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, 128, 128)*0.01) for _ in range(3)])
        self.range = 0.4
        self.shift = 0.1
        self.Nl = Nl
        self.Cl = Cl
        self.lines = nn.ParameterList([nn.Parameter(torch.rand(Nl, Cl)*self.range + self.shift, requires_grad=True) for _ in range(3)])
        # self.volume = nn.Parameter(torch.randn(1, Cv, Nv, Nv, Nv)*0.001)


        if not(convex):
            self.net = nn.Sequential(nn.Linear(Cl, 128), nn.ReLU(inplace=True), nn.Linear(128, output_dim))
        else:
            self.net = ConvexNet(in_features=Cl, hid_features=1024)

    def forward(self, coordinates):
        batch_size, n_coords, n_dims = coordinates.shape
        # print(coordinates.shape) # 1, 4096, 3])
        
        # x_max = torch.max(coordinates[..., 0])
        # x_min = torch.min(coordinates[..., 0])
        # y_max = torch.max(coordinates[..., 1])
        # y_min = torch.min(coordinates[..., 1])
        # z_max = torch.max(coordinates[..., 2])
        # z_min = torch.min(coordinates[..., 2])

        # x_ind =  torch.floor((coordinates[..., 0] - x_min) / (x_max - x_min) * (self.Nl-1)).to(int)
        # y_ind =  torch.floor((coordinates[..., 1] - y_min) / (y_max - y_min) * (self.Nl-1)).to(int)
        # z_ind =  torch.floor((coordinates[..., 2] - z_min) / (z_max - z_min) * (self.Nl-1)).to(int)
        # print((x_max - x_min))


        xcoord = coordinates[..., 0]
        ycoord = coordinates[..., 1]
        zcoord = coordinates[..., 2]
        D = 4
        k = torch.round((xcoord + D/2) * self.Nl/D).to(torch.int)
        l = torch.round((ycoord + D/2) * self.Nl/D).to(torch.int)
        m = torch.round((zcoord + D/2) * self.Nl/D).to(torch.int)

        cond = lambda x: torch.logical_and(x >= 0, x < self.Nl)

        cond_all = torch.logical_and(cond(k), torch.logical_and(cond(l), cond(m)))
        valid_all = torch.where(cond_all)

        k_valid = k[torch.where(cond(k))]
        l_valid = l[torch.where(cond(l))] #valid_all
        m_valid = m[torch.where(cond(m))]

        # print(k_valid.shape)
        # print(l_valid.shape)
        # print(m_valid.shape)

        # print(xcoord.device)
        x_embed = torch.zeros(n_coords, self.Cl).to(xcoord.device)
        y_embed = torch.zeros(n_coords, self.Cl).to(xcoord.device)
        z_embed = torch.zeros(n_coords, self.Cl).to(xcoord.device)

        x_embed[k_valid,:] = self.lines[0][k_valid,:]
        y_embed[l_valid,:] = self.lines[1][l_valid,:]
        z_embed[m_valid,:] = self.lines[2][m_valid,:]
        # print(self.lines[0])


        features = (x_embed + y_embed + z_embed)
        features2 = (x_embed * y_embed + x_embed * z_embed + y_embed * z_embed) * 1/self.range
        # print(features2.shape)
        # print(features2[0,0,:])
        features3 =  x_embed * y_embed * z_embed * 1/(self.range**2)
        # print(features.shape)
        # print(features3[0,0,:].shape)
        # print(features3[0,0,:])
        # features += x_embed * yz_embed + y_embed * xz_embed + z_embed * xy_embed
        # features += yz_embed + xz_embed + xy_embed
        # f = features + features2 + features3
        # print(torch.mean(f,dim=1).shape)
        # print(torch.mean(f,dim=1))
        # print(self.net(features + features2 + features3))
        return self.net(features + features2 + features3) #+ self.net2(features2) + self.net3(features3)


class TuckerPlanes(nn.Module):
    def __init__(self, convex=False, Cl=32, r=1.3, mode="combine", multiply=False):
        super().__init__()

        self.r = r
        self.mode = mode
        self.multiply = multiply
        self.range = 0.4
        self.shift = 0.1

        if self.mode == "useLines" or self.mode == "combine":
            self.lines = nn.ParameterList([nn.Parameter(torch.rand(1, Cl, 128, 1)*self.range + self.shift) for _ in range(3)])
        
        if self.mode == "usePlanes" or self.mode == "combine":
            self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, 128, 128)*self.range + self.shift) for _ in range(3)])     #*0.001) for _ in range(3)])


        self.Wx = nn.Linear(Cl, Cl, bias=False)
        self.Wy = nn.Linear(Cl, Cl, bias=False)
        self.Wz = nn.Linear(Cl, Cl, bias=False)
        if multiply:
            torch.nn.init.uniform_(self.Wx.weight, a=0.1, b=0.5)
            torch.nn.init.uniform_(self.Wy.weight, a=0.1, b=0.5)
            torch.nn.init.uniform_(self.Wz.weight, a=0.1, b=0.5)
        # self.alpha = nn.Linear(Cl, 1, bias=False)
        if not(convex):
            self.net = nn.Sequential(nn.Linear(Cl, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        else:
            self.net = ConvexNet(in_features=Cl, hid_features=256) # semi-semi convex


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
        

        if self.mode == "useLines" or self.mode == "combine":
            x_coords = coordinates[..., 0] / self.r
            y_coords = coordinates[..., 1] / self.r
            z_coords = coordinates[..., 2] / self.r
            ##### (x,x)
            xs = torch.stack([x_coords, x_coords],dim=-1)
            ys = torch.stack([y_coords, y_coords],dim=-1)
            zs = torch.stack([z_coords, z_coords],dim=-1)

            x_embed = self.sample_plane(xs, self.lines[0]).squeeze(0)
            y_embed = self.sample_plane(ys, self.lines[1]).squeeze(0)
            z_embed = self.sample_plane(zs, self.lines[2]).squeeze(0) # [batch, Cl]    

        if self.mode == "usePlanes" or self.mode == "combine":
            xy_embed = self.sample_plane(coordinates[..., 0:2] / self.r, self.embeddings[0]).squeeze(0)
            yz_embed = self.sample_plane(coordinates[..., 1:3] / self.r, self.embeddings[1]).squeeze(0)
            xz_embed = self.sample_plane(coordinates[..., :3:2] / self.r, self.embeddings[2]).squeeze(0) # [batch, Cl] 


        if self.mode == "useLines":

            xtran = self.Wx(x_embed)
            ytran = self.Wy(y_embed)
            ztran = self.Wz(z_embed)
            if self.multiply:
                features = xtran * ytran * ztran
            else:
                features = (x_embed * ytran + z_embed * xtran + y_embed * ztran)

        elif self.mode == "usePlanes":

            yztran = self.Wx(yz_embed)
            xztran = self.Wy(xz_embed)
            xytran = self.Wz(xy_embed)
            if self.multiply:
                features = xytran * yztran * xztran
            else:
                features = (xy_embed * yztran + yz_embed * xztran + xz_embed * xytran)
        
        elif self.mode == "combine":

            yztran = self.Wx(yz_embed)
            xztran = self.Wy(xz_embed)
            xytran = self.Wz(xy_embed)
            features = (x_embed * yztran + y_embed * xztran + z_embed * xytran)
        

        return self.net(features) 


class Tucker3D(nn.Module):
    def __init__(self, Cl=32, Nl=128, r=1.3, useLines=False):
        super().__init__()

        self.r = r
        self.range = 0.4
        self.shift = 0.1
        if useLines:
            self.lines = nn.ParameterList([nn.Parameter(torch.rand(1, Cl, Nl, 1)*self.range + self.shift) for _ in range(3)])
        else:
            self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, Nl, Nl)*0.001) for _ in range(3)])

        self.G = nn.Parameter(torch.rand(Cl, Cl, Cl) * self.range + self.shift)
        self.useLines = useLines

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
        
        if self.useLines:
            x_coords = coordinates[..., 0] / self.r
            y_coords = coordinates[..., 1] / self.r
            z_coords = coordinates[..., 2] / self.r
            ##### (x,x)
            xs = torch.stack([x_coords, x_coords],dim=-1)
            ys = torch.stack([y_coords, y_coords],dim=-1)
            zs = torch.stack([z_coords, z_coords],dim=-1)

            x_embed = self.sample_plane(xs, self.lines[0]).squeeze(0)
            y_embed = self.sample_plane(ys, self.lines[1]).squeeze(0)
            z_embed = self.sample_plane(zs, self.lines[2]).squeeze(0) # [batch, Cl]        
        else:
            xy_embed = self.sample_plane(coordinates[..., 0:2] / self.r, self.embeddings[0]).squeeze(0)
            yz_embed = self.sample_plane(coordinates[..., 1:3] / self.r, self.embeddings[1]).squeeze(0)
            xz_embed = self.sample_plane(coordinates[..., :3:2] / self.r, self.embeddings[2]).squeeze(0)

        if self.useLines:
            # xyz, nx, ny, nz -> n
            features = torch.einsum('xyz,nx,ny,nz->n', self.G,x_embed,y_embed,z_embed)
        else:
            features = torch.einsum('xyz,nx,ny,nz->n', self.G,xy_embed,yz_embed,xz_embed)
        return (features) 


class KernelPlanes(nn.Module):
    def __init__(self, linear=False, Cl=32, Nl=128, Cv=8, Nv=64, mode="combine", multiply=False, act="relu",lowres=True, split=False):
        super().__init__()

        self.mode = mode
        self.multiply = multiply
        self.range = 0.4
        self.shift = 0.1
        self.act = act
        self.split = split

        if self.mode == "useLines" or self.mode == "combine":
            self.lines = nn.ParameterList([nn.Parameter(torch.rand(1, Cl, Nl, 1)*self.range + self.shift) for _ in range(3)])
        
        if self.mode == "usePlanes" or self.mode == "combine":
            self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, Cl, Nl, Nl)*self.range + self.shift) for _ in range(3)])     #*0.001) for _ in range(3)])

        if lowres:
            self.volume = nn.Parameter(torch.randn(1, Cv, Nv, Nv, Nv)*self.range + self.shift)

        self.lowres = lowres

        # polynomial act
        self.phi = lambda x: x**2

        # mlp
        self.Phi_x = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(Cl, Cl))
        self.Phi_y = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(Cl, Cl))
        self.Phi_z = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(Cl, Cl))
        
        in_dim = Cl+Cv if (lowres and not(split)) else Cl

        if not(linear):
            self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
            if self.split:
                self.net2 = nn.Sequential(nn.Linear(Cv, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        else:
            self.net = nn.Linear(in_dim, 1)
            if self.split:
                self.net2 = nn.Sequential(nn.Linear(Cv, 1))


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
        

        if self.mode == "useLines" or self.mode == "combine":
            x_coords = coordinates[..., 0] 
            y_coords = coordinates[..., 1] 
            z_coords = coordinates[..., 2] 
            ##### (x,x)
            xs = torch.stack([x_coords, x_coords],dim=-1)
            ys = torch.stack([y_coords, y_coords],dim=-1)
            zs = torch.stack([z_coords, z_coords],dim=-1)

            x_embed = self.sample_plane(xs, self.lines[0]).squeeze(0)
            y_embed = self.sample_plane(ys, self.lines[1]).squeeze(0)
            z_embed = self.sample_plane(zs, self.lines[2]).squeeze(0) # [batch, Cl]    

        if self.mode == "usePlanes" or self.mode == "combine":
            xy_embed = self.sample_plane(coordinates[..., 0:2] , self.embeddings[0]).squeeze(0)
            yz_embed = self.sample_plane(coordinates[..., 1:3] , self.embeddings[1]).squeeze(0)
            xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2]).squeeze(0) # [batch, Cl] 

        if self.lowres:
            xyz_embed = self.sample_volume(coordinates, self.volume).squeeze(0)
            # print(xyz_embed.shape) [batch, Cv] 



        if self.mode == "useLines":

            if self.act == "relu":

                xtran = self.Phi_x(x_embed)
                ytran = self.Phi_y(y_embed)
                ztran = self.Phi_z(z_embed)
            elif self.act == "poly":

                xtran = self.phi(x_embed)
                ytran = self.phi(y_embed)
                ztran = self.phi(z_embed)
            else:
                xtran = (x_embed)
                ytran = (y_embed)
                ztran = (z_embed)

            if self.multiply:
                features = xtran * ytran * ztran
            else:
                features = (xtran + ytran + ztran)

        elif self.mode == "usePlanes":

            if self.act == "relu":
                yztran = self.Phi_x(yz_embed)
                xztran = self.Phi_y(xz_embed)
                xytran = self.Phi_z(xy_embed)
            elif self.act == "poly":
                yztran = self.phi(yz_embed)
                xztran = self.phi(xz_embed)
                xytran = self.phi(xy_embed)
            else:
                yztran = (yz_embed)
                xztran = (xz_embed)
                xytran = (xy_embed)
            
            if self.multiply:
                features = (xy_embed * yztran + yz_embed * xztran + xz_embed * xytran)
            else:
                features = xytran + yztran + xztran
        
        elif self.mode == "combine":

            if self.act == "relu":

                yztran = self.Phi_x(yz_embed)
                xztran = self.Phi_y(xz_embed)
                xytran = self.Phi_z(xy_embed)

                xtran = self.Phi_x(x_embed)
                ytran = self.Phi_y(y_embed)
                ztran = self.Phi_z(z_embed)
            elif self.act == "poly":

                yztran = self.phi(yz_embed)
                xztran = self.phi(xz_embed)
                xytran = self.phi(xy_embed)

                xtran = self.phi(x_embed)
                ytran = self.phi(y_embed)
                ztran = self.phi(z_embed)
            else:

                yztran = (yz_embed)
                xztran = (xz_embed)
                xytran = (xy_embed)

                xtran = (x_embed)
                ytran = (y_embed)
                ztran = (z_embed)

            features = (xtran * yztran + ytran * xztran + ztran * xytran) # could also split these terms and pass to net separately
        
        if self.lowres:
            
            if self.split:
                return self.net(features) + self.net2(xyz_embed)
            else:
                features = torch.cat([features, xyz_embed], dim=-1)
                # features += xyz_embed
                return self.net(features)

        return self.net(features) 


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

class TensorF(nn.Module):
    def __init__(self, Cl=32, Nl=128):
        super().__init__()

        self.range = 0.4
        self.shift = 0.1


        self.lines = nn.ParameterList([nn.Parameter(torch.rand(1, Cl, Nl, 1)*self.range + self.shift) for _ in range(3)])
        self.embeddings = nn.ParameterList([nn.Parameter(torch.rand(1, Cl, Nl, Nl)*self.range + self.shift) for _ in range(3)])     #*0.001) for _ in range(3)])
   
        in_dim = Cl
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))



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
        
        x_coords = coordinates[..., 0] 
        y_coords = coordinates[..., 1] 
        z_coords = coordinates[..., 2] 
        ##### (x,x)
        xs = torch.stack([x_coords, x_coords],dim=-1)
        ys = torch.stack([y_coords, y_coords],dim=-1)
        zs = torch.stack([z_coords, z_coords],dim=-1)

        x_embed = self.sample_plane(xs, self.lines[0]).squeeze(0)
        y_embed = self.sample_plane(ys, self.lines[1]).squeeze(0)
        z_embed = self.sample_plane(zs, self.lines[2]).squeeze(0) # [batch, Cl]    

        xy_embed = self.sample_plane(coordinates[..., 0:2] , self.embeddings[0]).squeeze(0)
        yz_embed = self.sample_plane(coordinates[..., 1:3] , self.embeddings[1]).squeeze(0)
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2]).squeeze(0) # [batch, Cl] 

        features = (x_embed * yz_embed + y_embed * xz_embed + z_embed * xy_embed) 

        return self.net(features) 
