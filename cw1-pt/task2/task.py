import torch
import torch.nn as nn
from myELM import MyExtremeLearningMachine
from myMixup import MyMixUp
from myEnsembleELM import MyEnsembleELM
from dataLoader import load_data
from utils import f1_score

def random_guess(test_loader, num_classes=10):
    correct = 0
    total = 0
    for _, labels in test_loader:
        outputs = torch.randint(0, num_classes, (labels.size(0),))
        total += labels.size(0)
        correct += (outputs == labels).sum().item()
    return correct / total

def basic_elm(train_loader, num_classes=10, lr=0.01, epochs=10):
    model = MyExtremeLearningMachine(3, 32, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

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
            # all_preds.extend(predicted.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        # f1 = f1_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
        # print(f'Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    return model

def mixed_elm(train_loader, is_mixup=False, is_ensemble=False, n_models=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选择模型
    if is_ensemble:
        model = MyEnsembleELM(
            n_models=n_models,
            input_channels=3,
            num_feature_maps=32,
            num_classes=10
        )
    else:
        model = MyExtremeLearningMachine(3, 256, 10, std=0.01)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    mixup = MyMixUp(alpha=0.4) if is_mixup else None

    for epoch in range(10):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if mixup:
                # Apply Mixup
                mixed_images, y_a, y_b, lam = mixup.mixup_data(images, labels)
                outputs = model(mixed_images)
                loss = mixup(outputs, y_a.long(), y_b.long(), lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f'Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

    return model



def save_model(model, filename):
    torch.save(model.state_dict(), f".model_saved/{filename}.pth")
    print(f"Model saved: {filename}.pth")

if __name__ == '__main__':
    train_loader, test_loader, classes = load_data()
    # save_model(basic_elm(train_loader), 'basic_elm')
    save_model(mixed_elm(train_loader, is_mixup=True), 'mixup_elm')
    save_model(mixed_elm(train_loader, is_ensemble=True), 'ensemble_elm')
    save_model(mixed_elm(train_loader, is_mixup=True, is_ensemble=True), 'mixup_ensemble_elm')


