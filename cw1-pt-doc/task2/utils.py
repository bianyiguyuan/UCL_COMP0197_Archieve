import torch
import os
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from PIL import ImageDraw, ImageFont


def f1_score(y_true, y_pred, num_classes=10):
    """
    Computes the weighted F1-score for a multi-class classification problem.

    Parameters:
    -----------
    y_true : list or torch.Tensor
        Ground-truth class labels.
    y_pred : list or torch.Tensor
        Predicted class labels.
    num_classes : int, optional
        Number of unique classes in the dataset (default is 10).

    Returns:
    --------
    float
        The weighted F1-score.
    """
    y_true = torch.tensor(y_true, dtype=torch.int64)  
    y_pred = torch.tensor(y_pred, dtype=torch.int64)

    class_counts = torch.zeros(num_classes, dtype=torch.float32)  
    f1_scores = torch.zeros(num_classes, dtype=torch.float32)  

    for class_idx in range(num_classes):
        true_positive = (y_pred.eq(class_idx) & y_true.eq(class_idx)).float().sum()
        false_positive = (y_pred.eq(class_idx) & y_true.ne(class_idx)).float().sum()
        false_negative = (y_pred.ne(class_idx) & y_true.eq(class_idx)).float().sum()
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores[class_idx] = f1
        class_counts[class_idx] = y_true.eq(class_idx).float().sum()

    total_samples = class_counts.sum().item()
    weighted_f1_score = (f1_scores * class_counts).sum().item() / (total_samples + 1e-8)

    return weighted_f1_score

def accuracy(y_pred, y_true):
    """
    Computes the accuracy of predictions.

    Parameters:
    -----------
    y_pred : list or np.ndarray
        Predicted class labels.
    y_true : list or np.ndarray
        Ground-truth class labels.

    Returns:
    --------
    float
        The classification accuracy.
    """
    y_pred = np.round(y_pred)  
    return np.mean(y_pred == y_true)

def evaluate_model(model, test_loader):
    """
    Evaluates the model on a test dataset and computes accuracy and F1-score.

    Parameters:
    -----------
    model : nn.Module
        The trained model to be evaluated.
    test_loader : DataLoader
        The DataLoader containing the test dataset.

    Returns:
    --------
    tuple
        Accuracy and weighted F1-score of the model on the test dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds) 
    return acc, f1


def visualize_results(model, test_loader, classes, pic_name, images_shown=36):
    """
    Visualizes the model's predictions on test images and saves the results as an image.

    Parameters:
    -----------
    model : nn.Module
        The trained model for visualization.
    test_loader : DataLoader
        The DataLoader containing the test dataset.
    classes : list
        A list of class names corresponding to dataset labels.
    pic_name : str
        The filename to save the visualization image.
    images_shown : int, optional
        Number of test images to visualize (default is 36).

    Returns:
    --------
    None
        Saves the image with predicted and ground-truth labels.
    """
    model.eval()

    images = []
    labels = []
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            images.append(batch_images)
            labels.append(batch_labels)
            if len(images) * batch_images.size(0) >= images_shown:  
                break

    images = torch.cat(images, dim=0)[:images_shown]
    labels = torch.cat(labels, dim=0)[:images_shown]

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)  
    grid_img = vutils.make_grid(images, nrow=6, padding=10, scale_each=True)
    np_img = grid_img.permute(1, 2, 0).cpu().numpy() 
    np_img = (np_img * 255).astype("uint8")  
    img_pil = Image.fromarray(np_img)  
    img_pil = img_pil.resize((900, 900), Image.LANCZOS)

    img_width, img_height = img_pil.size
    single_img_width = img_width // 6
    single_img_height = img_height // 6
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)  
    except IOError:
        font = ImageFont.load_default()

    for i in range(images_shown):
        row, col = divmod(i, 6)
        x = col * single_img_width + 30  
        y = (row + 1) * single_img_height - 50  
        gt_class = classes[labels[i].item()]
        pred_class = classes[predicted[i].item()]
        text_str = f"GT: {gt_class}\nPred: {pred_class}"
        draw.text((x, y), text_str, fill=(255, 0, 0), font=font)

    script_dir = os.path.dirname(os.path.abspath(__file__))  
    save_path = os.path.join(script_dir, pic_name)
    img_pil.save(save_path)
    print(f"Visualization saved to '{save_path}'")