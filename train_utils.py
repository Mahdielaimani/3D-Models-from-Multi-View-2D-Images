import torch
from torch.utils.data import DataLoader
import os
import logging
from custom_dataset import MultiViewAirplaneDataset
from model_3d import AirplaneGenerator3D, create_airplane_template, voxel_loss
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import trimesh
import skimage.measure
from torch.optim.lr_scheduler import ReduceLROnPlateau

def voxel_to_obj(voxels, threshold=0.5, scale=0.1):
    """Convert voxel data to OBJ format with improved detail"""
    vertices = []
    faces = []
    vert_index = 1

    for x in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for z in range(voxels.shape[2]):
                if voxels[x, y, z] > threshold:
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            for dz in [0, 1]:
                                vertices.append(((x+dx)*scale, (y+dy)*scale, (z+dz)*scale))
                    
                    faces.extend([
                        (vert_index, vert_index+1, vert_index+3, vert_index+2),
                        (vert_index+4, vert_index+5, vert_index+7, vert_index+6),
                        (vert_index, vert_index+2, vert_index+6, vert_index+4),
                        (vert_index+1, vert_index+3, vert_index+7, vert_index+5),
                        (vert_index, vert_index+1, vert_index+5, vert_index+4),
                        (vert_index+2, vert_index+3, vert_index+7, vert_index+6)
                    ])
                    vert_index += 8

    obj_lines = ['# OBJ file']
    for v in vertices:
        obj_lines.append(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}')
    for f in faces:
        obj_lines.append(f'f {f[0]} {f[1]} {f[2]} {f[3]}')

    return '\n'.join(obj_lines)

def generate_3d_mesh(voxels, threshold=0.5, scale=0.1):
    """Generate a 3D mesh from voxel data using trimesh"""
    voxels_binary = (voxels > threshold).astype(np.uint8)
    vertices, faces, _, _ = skimage.measure.marching_cubes(voxels_binary, level=0.5)
    
    vertices *= scale
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def generate_and_save_3d_model(model, test_dataset, epoch, config):
    model.eval()
    device = torch.device(config['device'])
    
    test_sample = random.choice(test_dataset)
    test_input = test_sample.unsqueeze(0).to(device) 
    
    with torch.no_grad():
        voxels = model(test_input)
    
    voxels_np = voxels.squeeze().cpu().numpy()
    
    obj_data = voxel_to_obj(voxels_np, threshold=0.3, scale=0.1)
    obj_path = os.path.join(config['output_dir'], f'airplane_3d_epoch_{epoch}.obj')
    with open(obj_path, 'w') as f:
        f.write(obj_data)
    
    mesh = generate_3d_mesh(voxels_np, threshold=0.3, scale=0.1)
    stl_path = os.path.join(config['output_dir'], f'airplane_3d_epoch_{epoch}.stl')
    mesh.export(stl_path)
    
    visualize_3d_model(voxels_np, epoch, config['output_dir'])
    print(f"Generated and saved 3D airplane model for epoch {epoch}")

def visualize_3d_model(voxel_data, epoch, output_dir, threshold=0.3):
    """Visualize the 3D model with improved visibility"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.where(voxel_data > threshold)
    
    if len(x) == 0:
        print(f"Warning: No points above threshold {threshold}. Adjusting threshold...")
        threshold = np.percentile(voxel_data, 95)  
        x, y, z = np.where(voxel_data > threshold)
    
    scatter = ax.scatter(x, y, z, 
                        c=voxel_data[x, y, z],
                        cmap='viridis',
                        alpha=0.8,          
                        s=30)             

   
    plt.colorbar(scatter)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Airplane Model - Epoch {epoch}')

    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    
    ax.view_init(elev=30, azim=45)

    ax.grid(False)

    plt.savefig(os.path.join(output_dir, f'airplane_3d_epoch_{epoch}.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close(fig)

def setup_training(config):
    full_dataset = MultiViewAirplaneDataset(config['dataset_path'])
    
    # (80% train, 10% val, 10% test)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = (total_size - train_size) // 2
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"\nDataset split:")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_dataset)} (80%)")
    print(f"Validation samples: {len(val_dataset)} (10%)")
    print(f"Test samples: {len(test_dataset)} (10%)")
    print()

    return train_loader, val_loader, test_dataset

def train_loop(train_loader, val_loader, test_dataset, config):
    device = torch.device(config['device'])
    model = AirplaneGenerator3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                 factor=0.5, patience=5, 
                                 verbose=True)
    
    best_loss = float('inf')
    current_lr = get_lr()
    
    airplane_template = create_airplane_template().to(device)
    
    print("\n" + "="*50)
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training on: {config['device']}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Initial learning rate: {current_lr}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("="*50 + "\n")
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        
        for batch_idx, batch in train_pbar:
            images = batch.to(device)
            batch_size = images.size(0)
            
            optimizer.zero_grad()
            voxels = model(images)
            
            loss = voxel_loss(voxels, airplane_template.expand(batch_size, 64, 64, 64), airplane_template)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item() * batch_size
            num_batches += batch_size
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / num_batches
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}, Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                        desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
        
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                images = batch.to(device)
                batch_size = images.size(0)
                voxels = model(images)
                
                loss = voxel_loss(voxels, airplane_template.expand(batch_size, 64, 64, 64), airplane_template)
                total_val_loss += loss.item() * batch_size
                num_val_batches += batch_size
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / num_val_batches
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}, Validation Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        current_lr = get_lr()
        print(f"Current learning rate: {current_lr}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config["checkpoint_path"] + ".best")
            print(f"New best model saved! Loss: {avg_val_loss:.4f}")
        
        if (epoch + 1) % config['save_every'] == 0:
            generate_and_save_3d_model(model, test_dataset, epoch + 1, config)
    
    return model

