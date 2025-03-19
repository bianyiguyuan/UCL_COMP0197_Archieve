import torch

def f1_score(y_true, y_pred, num_classes=10):
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
