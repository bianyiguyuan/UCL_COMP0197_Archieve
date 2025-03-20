import time
import torch
import random
import numpy as np
import os
from dataLoader import load_data
from myELM import MyExtremeLearningMachine
from modelTool import save_model, load_model
from utils import visualize_results, evaluate_model

def evaluate(model, data_loader):
    """
    Evaluates the model performance on a given dataset.

    Parameters:
    -----------
    model : nn.Module
        The trained model to be evaluated.
    data_loader : DataLoader
        The DataLoader containing the dataset for evaluation.

    Returns:
    --------
    float
        The accuracy of the model on the provided dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def part_one():
    """
    Trains and evaluates two models using Stochastic Gradient Descent (SGD) and Least Squares (LS) optimization.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
        Saves the trained models and prints training time and accuracy.
    """
    train_loader, test_loader, classes = load_data()
    model_sgd = MyExtremeLearningMachine(3, 32, 10)
    start = time.time()
    model_sgd.fit_elm_sgd(train_loader, lr=0.01, epochs=10)
    end = time.time()
    sgd_time = end - start
    sgd_acc = evaluate(model_sgd, test_loader)
    print(f"[SGD] Training Time: {sgd_time:.2f}s, Test Accuracy: {sgd_acc:.4f}")
    save_model(model_sgd, "basic_sgd", "model_2a")
    

    model_ls = MyExtremeLearningMachine(3, 32, 10)
    start = time.time()
    model_ls.fit_elm_ls(train_loader, lam=1e-2)
    end = time.time()
    ls_time = end - start
    ls_acc = evaluate(model_ls, test_loader)
    print(f"[LS] Training Time: {ls_time:.2f}s, Test Accuracy: {ls_acc:.4f}")
    save_model(model_ls, "basic_ls", "model_2a")

def random_hyperparameter_search(model_class, train_loader, test_loader, search_space):
    """
    Performs a random hyperparameter search to find the best-performing model.

    Parameters:
    -----------
    model_class : class
        The model class (e.g., MyExtremeLearningMachine).
    train_loader : DataLoader
        The DataLoader containing the training dataset.
    test_loader : DataLoader
        The DataLoader containing the test dataset.
    search_space : dict
        The dictionary specifying the range of hyperparameters to search.

    Returns:
    --------
    tuple
        The best hyperparameters found and the trained model with the best performance.
    """
    best_acc = 0
    best_params = None
    best_model = ""
    
    for _ in range(20):
        params = {
            "lam": round(float(random.choice(search_space["lam"])), 5),
            "std": round(float(random.choice(search_space["std"])), 5),
        }

        model = model_class(3, 32, 10, std=params["std"])
        model.fit_elm_ls(train_loader, params["lam"])
        
        acc = evaluate(model, test_loader)
        print(f"Test Accuracy: {acc:.4f} | Params: {params}")
        save_model(model, f"std{params["std"]}_lam{params["lam"]}_ls", "model_2a")

        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_model = f"std{params["std"]}_lam{params["lam"]}_ls"

    print(f"Best Params: {best_params}, Best Accuracy: {best_acc:.4f}")
    final_best_model = load_model(best_model, "model_2a")
    return best_params, final_best_model


def read_log(log_dir="compair_log.txt"):
    """
    Reads and prints the contents of a training log file.

    Parameters:
    -----------
    log_dir : str, optional
        The file name of the log file (default is "compair_log.txt").

    Returns:
    --------
    None
        Prints the log content if found; otherwise, prints a warning.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))  
    log_file = os.path.join(cur_dir, log_dir)
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                print(line.strip())
    else:
        print("No training log found!")


if __name__ == "__main__":
    TRAIN_MODE = False
    TEST_MODE = True

    train_loader, test_loader, classes = load_data()

    if TRAIN_MODE:
        part_one()
        search_space = {
            "lam": np.logspace(-3, -1, num=10),
            "std": np.linspace(0.0005, 0.05, num=10)
        }
        best_params, final_best_model = random_hyperparameter_search(MyExtremeLearningMachine, train_loader, test_loader, search_space)

    if TEST_MODE:
        print("")
        read_log()
        final_best_model = load_model("std0.0005_lam0.00464_ls", "model_2a")
        accuracy, f1 = evaluate_model(final_best_model, test_loader)
        print("\nBest LS param by random search:")
        print("std:0.0005 lambda:0.00464")
        print(f"\nBest Random Search Model Performance:")
        print(f"{'Metric':<15}{'Value':<10}")
        print(f"{'Accuracy':<15}{accuracy:<10.4f}")
        print(f"{'F1-score':<15}{f1:<10.4f}")
        visualize_results(final_best_model, test_loader, classes, "new_result.png")


