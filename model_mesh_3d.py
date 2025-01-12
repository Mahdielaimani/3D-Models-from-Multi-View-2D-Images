import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch3d.structures
import pytorch3d.renderer

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x):
        return self.features(x)

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        
    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)  # (seq_len, batch, features)
        x = self.transformer(x)
        return x.permute(1, 2, 0).view(x.size(1), -1)  # (batch, features)

class MeshGenerator(nn.Module):
    def __init__(self, input_dim, num_vertices=2500, num_faces=4996):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc_vertices = nn.Linear(512, num_vertices * 3)
        self.fc_faces = nn.Linear(512, num_faces * 3)
        
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        vertices = self.fc_vertices(x).view(-1, self.num_vertices, 3)
        faces = self.fc_faces(x).view(-1, self.num_faces, 3)
        faces = F.softmax(faces, dim=2)  
        return vertices, faces

class AirplaneGenerator3D(nn.Module):
    def __init__(self, num_views=13):  
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.fpn = FeaturePyramidNetwork()
        self.transformer = TransformerEncoder()
        self.mesh_generator = MeshGenerator(input_dim=256)
        
    def forward(self, x):
        batch_size, views, c, h, w = x.shape
        x = x.view(batch_size * views, c, h, w)
        features = self.feature_extractor(x)
        features = self.fpn(features)
        features = features.view(batch_size, views, *features.shape[1:])
        features = self.transformer(features.mean(dim=1))  
        vertices, faces = self.mesh_generator(features)
        return vertices, faces

class DifferentiableRenderer(nn.Module):
    def __init__(self, image_size=224, device='cuda'):
        super().__init__()
        self.device = device
        self.renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=pytorch3d.renderer.FoVPerspectiveCameras(device=device),
                raster_settings=pytorch3d.renderer.RasterizationSettings(
                    image_size=image_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                )
            ),
            shader=pytorch3d.renderer.SoftPhongShader(device=device)
        )

    def forward(self, vertices, faces, rotation):
        batch_size = vertices.shape[0]
        mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces)
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=rotation,
            T=[[0, 0, 2]] * batch_size,
            device=self.device
        )
        
        images = self.renderer(mesh, cameras=cameras)
        return images

def silhouette_loss(rendered_images, target_images):
    rendered_silhouette = (rendered_images.sum(dim=-1) > 0).float()
    target_silhouette = (target_images.sum(dim=-1) > 0).float()
    return F.binary_cross_entropy(rendered_silhouette, target_silhouette)

def symmetry_loss(vertices):
    left_half = vertices[:, :, 0] > 0
    right_half = vertices[:, :, 0] < 0
    left_vertices = vertices[left_half]
    right_vertices = vertices[right_half]
    right_vertices_flipped = right_vertices.clone()
    right_vertices_flipped[:, 0] *= -1
    return F.mse_loss(left_vertices, right_vertices_flipped)

def smoothness_loss(vertices, faces):
    v0 = torch.index_select(vertices, 1, faces[:, :, 0])
    v1 = torch.index_select(vertices, 1, faces[:, :, 1])
    v2 = torch.index_select(vertices, 1, faces[:, :, 2])
    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2
    
    normal = torch.cross(e01, e12)
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
    
    cosine = torch.sum(normal[:, :-1] * normal[:, 1:], dim=-1)
    return torch.mean(1 - cosine)

def chamfer_distance(pred_vertices, gt_vertices):
    dist_pred_gt = torch.cdist(pred_vertices, gt_vertices)
    dist_gt_pred = torch.cdist(gt_vertices, pred_vertices)
    
    min_dist_pred_gt, _ = torch.min(dist_pred_gt, dim=2)
    min_dist_gt_pred, _ = torch.min(dist_gt_pred, dim=2)
    
    chamfer_dist = torch.mean(min_dist_pred_gt) + torch.mean(min_dist_gt_pred)
    return chamfer_dist

def normal_consistency_loss(vertices, faces):
    v0 = torch.index_select(vertices, 1, faces[:, :, 0])
    v1 = torch.index_select(vertices, 1, faces[:, :, 1])
    v2 = torch.index_select(vertices, 1, faces[:, :, 2])
    
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (torch.norm(face_normals, dim=-1, keepdim=True) + 1e-6)
    
    consistency = torch.mean(torch.abs(face_normals[:, :-1] - face_normals[:, 1:]))
    return consistency

def edge_length_regularization(vertices, faces):
    v0 = torch.index_select(vertices, 1, faces[:, :, 0])
    v1 = torch.index_select(vertices, 1, faces[:, :, 1])
    v2 = torch.index_select(vertices, 1, faces[:, :, 2])
    
    e01 = torch.norm(v1 - v0, dim=-1)
    e12 = torch.norm(v2 - v1, dim=-1)
    e20 = torch.norm(v0 - v2, dim=-1)
    
    edge_lengths = torch.cat([e01, e12, e20], dim=-1)
    mean_length = torch.mean(edge_lengths)
    
    return torch.mean((edge_lengths - mean_length)**2)

def mesh_loss(vertices, faces, target_images, renderer, gt_vertices=None):
    rotations = [
        pytorch3d.renderer.rotation_conversions.euler_angles_to_matrix(
            torch.tensor([0, angle, 0]), "XYZ"
        ) for angle in torch.linspace(0, 2*torch.pi, steps=13)  
    ]
    rotations = torch.stack(rotations).to(vertices.device)
    
    total_loss = 0
    for i, R in enumerate(rotations):
        rendered_images = renderer(vertices, faces, R.unsqueeze(0).expand(vertices.shape[0], -1, -1))
        total_loss += silhouette_loss(rendered_images, target_images[:, i])
    
    total_loss += 0.1 * symmetry_loss(vertices)
    total_loss += 0.05 * smoothness_loss(vertices, faces)
    total_loss += 0.1 * normal_consistency_loss(vertices, faces)
    total_loss += 0.01 * edge_length_regularization(vertices, faces)
    
    if gt_vertices is not None:
        total_loss += 0.1 * chamfer_distance(vertices, gt_vertices)
    
    return total_loss

