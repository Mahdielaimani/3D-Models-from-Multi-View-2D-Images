import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AirplaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AirplaneDecoder(nn.Module):
    def __init__(self, latent_dim=256, voxel_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512 * 4 * 4 * 4)
        self.deconv1 = nn.ConvTranspose3d(512, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose3d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose3d(128, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose3d(64, 1, 4, 2, 1)
        
        self.bn1 = nn.BatchNorm3d(256)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(64)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 512, 4, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        return x

class AirplaneGenerator3D(nn.Module):
    def __init__(self, num_views=14, latent_dim=256, voxel_dim=64):
        super().__init__()
        self.encoder = AirplaneEncoder()
        self.decoder = AirplaneDecoder(latent_dim, voxel_dim)
        self.fusion = nn.Linear(num_views * latent_dim, latent_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size, num_views, c, h, w = x.size()
        x = x.view(batch_size * num_views, c, h, w)
        features = self.encoder(x)
        features = features.view(batch_size, -1)
        features = self.dropout(features)
        fused = self.fusion(features)
        voxels = self.decoder(fused)
        return voxels.squeeze(1)

def create_airplane_template(voxel_dim=64):
    template = torch.zeros((voxel_dim, voxel_dim, voxel_dim))
    
    fuselage_length = int(voxel_dim * 0.8)
    fuselage_width = int(voxel_dim * 0.15)
    fuselage_height = int(voxel_dim * 0.15)
    start = (voxel_dim - fuselage_length) // 2
    template[start:start+fuselage_length, 
             voxel_dim//2-fuselage_width//2:voxel_dim//2+fuselage_width//2, 
             voxel_dim//2-fuselage_height//2:voxel_dim//2+fuselage_height//2] = 1

    wing_length = int(voxel_dim * 0.6)
    wing_width = int(voxel_dim * 0.5)
    wing_height = int(voxel_dim * 0.05)
    wing_start = voxel_dim // 2 - wing_length // 2
    wing_y_pos = voxel_dim // 2
    template[wing_start:wing_start+wing_length, 
             wing_y_pos-wing_width//2:wing_y_pos+wing_width//2, 
             voxel_dim//2-wing_height//2:voxel_dim//2+wing_height//2] = 1

    tail_length = int(voxel_dim * 0.2)
    tail_width = int(voxel_dim * 0.1)
    tail_height = int(voxel_dim * 0.2)
    tail_start = start + fuselage_length - tail_length
    template[tail_start:tail_start+tail_length, 
             voxel_dim//2-tail_width//2:voxel_dim//2+tail_width//2, 
             voxel_dim//2:voxel_dim//2+tail_height] = 1

    v_stab_length = int(voxel_dim * 0.15)
    v_stab_width = int(voxel_dim * 0.05)
    v_stab_height = int(voxel_dim * 0.15)
    v_stab_start = tail_start
    template[v_stab_start:v_stab_start+v_stab_length, 
             voxel_dim//2-v_stab_width//2:voxel_dim//2+v_stab_width//2, 
             voxel_dim//2:voxel_dim//2+v_stab_height] = 1

    return template

def voxel_loss(predictions, targets, template):
    bce_loss = F.binary_cross_entropy(predictions, targets)
    
    template_loss = F.binary_cross_entropy(predictions, template.expand_as(predictions))
    
    smoothness_loss = torch.mean(torch.abs(predictions[:, 1:] - predictions[:, :-1])) + \
                     torch.mean(torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1])) + \
                     torch.mean(torch.abs(predictions[:, :, :, 1:] - predictions[:, :, :, :-1]))
    
    symmetry_loss = torch.mean(torch.abs(
        predictions[:, :, :predictions.size(2)//2, :] - 
        torch.flip(predictions[:, :, predictions.size(2)//2:, :], [2])
    ))
    
    total_loss = bce_loss + 0.1 * template_loss + 0.05 * smoothness_loss + 0.1 * symmetry_loss
    
    return total_loss

