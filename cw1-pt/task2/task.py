import torch
import os
import torch.nn as nn
from myELM import MyExtremeLearningMachine
from myMixup import MyMixUp
from myEnsembleELM import MyEnsembleELM
from dataLoader import load_data
from utils import f1_score, evaluate_model, visualize_results
from modelTool import load_model, save_model
from logTool import read_log, clear_log, write_log


def random_guess(test_loader, num_classes=10):
    correct = 0
    total = 0
    for _, labels in test_loader:
        outputs = torch.randint(0, num_classes, (labels.size(0),))
        total += labels.size(0)
        correct += (outputs == labels).sum().item()

    rand_acc = correct / total
    print(f"Random Guess Accuracy: {rand_acc:.4f} (Expected ~{1/len(classes):.4f})")


def basic_elm(train_loader, num_classes=10, lr=0.01, epochs=10):
    model = MyExtremeLearningMachine(3, 32, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    write_log("\n=== Training basic ELM ===")

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        write_log(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")


    return model

def mixed_elm(train_loader, is_mixup=False, is_ensemble=False, n_models=5):
    if is_ensemble:
        model = MyEnsembleELM(
            n_models=n_models,
            input_channels=3,
            num_feature_maps=32,
            num_classes=10
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    else:
        model = MyExtremeLearningMachine(3, 32, 10, std=0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    mixup = MyMixUp(alpha=0.1) if is_mixup else None

    write_log(f"=== Training {'Ensemble' if is_ensemble else 'Basic ELM'} Model {'with Mixup' if is_mixup else ''} ===")

    for epoch in range(10):
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            optimizer.zero_grad()

            if mixup:
                mixed_images, targets_a, targets_b, lam = mixup.mixup_data(images, labels)
                outputs = model(mixed_images)
                loss = mixup.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (lam * predicted.eq(targets_a).sum().item() +
                            (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

        write_log(f"Epoch {epoch+1}/10 | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

    return model



if __name__ == '__main__':
    TRAIN_MODE = False
    TEST_MODE = True

    train_loader, test_loader, classes = load_data()
    print("Number of classes:", len(classes))

    if TRAIN_MODE:
        clear_log()

        print("\nRandom Guessing Baseline...")
        random_guess(test_loader)

        print("*" * 120)
        print("Training basic ELM model...")
        save_model(basic_elm(train_loader), 'basic_elm')

        print("*" * 120)
        print("Training ELM model with mixup...")
        save_model(mixed_elm(train_loader, is_mixup=True), 'mixup_elm')

        print("*" * 120)
        print("Training ensembled ELM model...")
        save_model(mixed_elm(train_loader, is_ensemble=True), 'ensemble_elm')

        print("*" * 120)
        print("Training ensembled ELM model with mixup...")
        save_model(mixed_elm(train_loader, is_mixup=True, is_ensemble=True), 'mixup_ensemble_elm')

    if TEST_MODE:
        read_log()

        print("\nRandom Guessing Baseline...")
        random_guess(test_loader)

        print("\nRandom guessing means selecting a class uniformly at random without considering input features.")
        print("Its expected accuracy is 1/C, where C is the number of classes.")
        print("To test, generate random labels using torch.randint(0, C, size) and compute accuracy.")
        print("If the result is close to 1/C, it confirms random guessing.")


        print("\nEvaluating models on test set...")

        models = {
            "basic_elm": load_model("basic_elm"),
            "mixup_elm": load_model("mixup_elm"),
            "ensemble_elm": load_model("ensemble_elm", is_ensemble=True),
            "mixup_ensemble_elm": load_model("mixup_ensemble_elm", is_ensemble=True),
        }
        
        print(f"\n{'Model':<25}{'Accuracy':<12}{'F1-score':<12}")
        best_model = None
        best_acc = 0

        for model_name, model in models.items():
            acc, f1 = evaluate_model(model, test_loader)
            print(f"{model_name:<25}{acc:<12.4f}{f1:<12.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = model_name

        print(f"\nBest Model: {best_model} with Accuracy: {best_acc:.4f}")

        final_best_model = models[best_model]
        visualize_results(final_best_model, test_loader, classes, "result.png")