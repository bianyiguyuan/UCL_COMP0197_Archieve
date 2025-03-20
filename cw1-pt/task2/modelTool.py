import torch
import os
from myELM import MyExtremeLearningMachine
from myEnsembleELM import MyEnsembleELM

def save_model(model, filename, folder_dir=None):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    if folder_dir:
        save_folder = os.path.join(current_file_dir, folder_dir)
    else:
        save_folder = current_file_dir 
    os.makedirs(save_folder, exist_ok=True)  
    save_path = os.path.join(save_folder, f"{filename}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model {filename} saved")

def load_model(filename, folder_dir=None, is_ensemble=False, n_models=5):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    if folder_dir:
        load_folder = os.path.join(current_file_dir, folder_dir)
    else:
        load_folder = current_file_dir

    load_path = os.path.join(load_folder, f"{filename}.pth")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file {load_path} not found!")

    if is_ensemble:
        model = MyEnsembleELM(
            n_models=n_models,
            input_channels=3,
            num_feature_maps=32,
            num_classes=10
        )
    else:
        model = MyExtremeLearningMachine(3, 32, 10)

    state_dict = torch.load(load_path, weights_only=True)

    if is_ensemble and "models.0.conv_layer.weight" in state_dict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict)
    model.eval()  
    print(f"Model '{filename}' loaded successfully")

    return model