import torch
from myELM import MyExtrmeLearningMachine
from myMixup import MyMixUp
from myEnsembleELM import MyEnsembleELM
from dataLoader import load_data

def random_guess(test_loader, num_classes=10):
    correct = 0
    total = 0
    for _, labels in test_loader:
        outputs = torch.randint(0, num_classes, (labels.size(0),))
        total += labels.size(0)
        correct += (outputs == labels).sum().item()
    return correct / total

def basic_elm(train_loader, num_classes=10, lr=0.01, epochs=10):
    model = MyExtrmeLearningMachine(3, 32, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

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
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {accuracy}')
    return model

def mixed_elm(train_loader, is_mixup=False, is_ensemble=False, n_models=5):
    if is_ensemble:
        model = MyEnsembleELM(
        n_models=5,
        input_channels=3,
        num_feature_maps=32,
        num_classes=10
    )
    else:
        model = MyExtrmeLearningMachine(3, 32, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    if is_mixup:
        mixup = MyMixUp(alpha=1.0)
    else:
        mixup = None

    for epoch in range(10):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if mixup:
                index, lam = mixup.mixup_indices(images)
                mixed_images = lam * images + (1 - lam) * images[index]
                y_a, y_b = labels, labels[index]
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        print(f'Epoch {epoch+1}/10, Loss: {epoch_loss}, Accuracy: {accuracy}')
    return model

def save_model(model, filename):
    torch.save(model.state_dict(), f"cw1-pt/task2/{filename}.pth")
    print(f"Model saved: {filename}.pth")

if __name__ == '__main__':
    train_loader, test_loader, classes = load_data()
    # save_model(basic_elm(train_loader), 'basic_elm')
    save_model(mixed_elm(train_loader, is_mixup=True), 'mixup_elm')
    save_model(mixed_elm(train_loader, is_ensemble=True), 'ensemble_elm')
    save_model(mixed_elm(train_loader, is_mixup=True, is_ensemble=True), 'mixup_ensemble_elm')


