import os
import torch
import logging
from datetime import datetime
from train_utils import setup_training, train_loop
from model_3d import AirplaneGenerator3D
import numpy as np
from PIL import Image

def get_memory_usage():
    import psutil
    cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 
    return cpu_memory, gpu_memory

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting main function")
    logging.info(f"Current time: {datetime.now().strftime('%c')}")
    
    print("Current working directory:", os.getcwd())
    
    WORKING_DIR = os.getcwd()
    logging.info(f"Working directory: {WORKING_DIR}")
    
    print("Contents of working directory:", os.listdir(WORKING_DIR))
    
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "batch_size": 32,
        "save_every": 10,
        "output_dir": os.path.join(WORKING_DIR, "output"),
        "log_dir": os.path.join(WORKING_DIR, "logs"),
        "checkpoint_path": os.path.join(WORKING_DIR, "checkpoints", "airplane_generator_3d.pth"),
        "dataset_path": os.path.join(WORKING_DIR, "dataset", "dataset"),
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)

    logging.info(f"Dataset path: {config['dataset_path']}")
    logging.info(f"Directory exists: {os.path.exists(config['dataset_path'])}")
    if os.path.exists(config['dataset_path']):
        logging.info(f"Contents of dataset directory: {os.listdir(config['dataset_path'])}")

    try:
        logging.info("Setting up training components")
        train_loader, val_loader, test_dataset = setup_training(config)
        logging.info("Training components set up successfully")
        
        cpu_mem, gpu_mem = get_memory_usage()
        logging.info(f"CPU Memory usage: {cpu_mem:.2f} MB")
        logging.info(f"GPU Memory usage: {gpu_mem:.2f} MB")

        logging.info("Starting training loop")
        model = train_loop(train_loader, val_loader, test_dataset, config)
        
        logging.info("Training completed successfully")

        torch.save(model.state_dict(), config["checkpoint_path"])
        logging.info(f"Final model saved to {config['checkpoint_path']}")

        cpu_mem, gpu_mem = get_memory_usage()
        logging.info(f"Final CPU Memory usage: {cpu_mem:.2f} MB")
        logging.info(f"Final GPU Memory usage: {gpu_mem:.2f} MB")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.error(f"Current directory contents: {os.listdir(os.getcwd())}")
        raise

if __name__ == "__main__":
    main()

