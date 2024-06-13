import os
import time

import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from MozillaCommonVoiceDatasetAST import get_dataloader, get_labels, ids2labels, get_num_labels, get_classes
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, classification_report


def main():
    train_metadata = r"../../data/metadata/train.csv"
    valid_metadata = r"../../data/metadata/val.csv"
    test_metadata = r"../../data/metadata/test.csv"
    data_dir = r"../../data/clips/"

    model_save_dir = "../../models/ast/"
    model_to_load = "../../models/ast/ast_epoch_10"

    results_dir = "../../data/plots/"

    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-5

    train(train_metadata, valid_metadata, data_dir, num_epochs=num_epochs, batch_size=batch_size,
          learning_rate=learning_rate,
          save_dir=model_save_dir)
    test(test_metadata, data_dir, results_dir, model_to_load, batch_size=batch_size)
    time_test(test_metadata, data_dir, model_to_load, batch_size=batch_size)


def train(train_metadata, valid_metadata, data_dir, num_epochs=5, batch_size=4, learning_rate=1e-5, save_dir="models/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    print(f"Training model for {num_epochs} epochs")

    label2id = get_labels(train_metadata)
    id2label = ids2labels(label2id)
    num_labels = get_num_labels(train_metadata)

    # Load the pre-trained model
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",
                                                      num_labels=num_labels, id2label=id2label, label2id=label2id,
                                                      ignore_mismatched_sizes=True).to(device)
    # Load the feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Load the dataset
    train_dataloader = get_dataloader(train_metadata, data_dir, batch_size=batch_size, is_augment=True)
    valid_dataloader = get_dataloader(valid_metadata, data_dir, batch_size=batch_size)

    # Define the optimizer, loss function and learning rate scheduler
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs + 1, eta_min=1e-10)

    # Fine-tune the model
    for epoch in range(num_epochs):
        model.train()
        epoch_correct = 0
        total = 0

        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for inputs, targets in pbar:
            inputs = inputs.squeeze().to('cpu').numpy()

            inputs = feature_extractor(inputs, sampling_rate=16000, padding="max_length",
                                       return_tensors="pt").input_values

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.softmax(outputs.logits, dim=1).max(1)
            epoch_correct += torch.eq(predicted, targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix_str(f"Batch Loss: {loss.item():.4f} Epoch Accuracy: {(epoch_correct / total) * 100:.2f}%")
            writer.add_scalar("Loss/train", loss.item(), epoch)

        print(f"Epoch {epoch + 1} completed")

        writer.add_scalar("Accuracy/train", epoch_correct / total, epoch)
        print(f"Train Accuracy: {(epoch_correct / total) * 100:.2f}%")

        # update the learning rate
        scheduler.step()

        # save the model after each epoch
        model.save_pretrained(os.path.join(save_dir, f"ast_epoch_{epoch + 1}"))

        print("Calculating validation metrics...")
        val_accuracy, val_loss = calculate_metrics(valid_dataloader, model, feature_extractor, criterion, device)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"Validation Loss: {val_loss:.4f} Validation Accuracy: {val_accuracy:.2f}")


def calculate_metrics(loader, model, feature_extractor, criterion, device):
    model.eval()

    correct = 0
    total = 0
    loss = 0

    with torch.inference_mode():
        pbar = tqdm.tqdm(loader, desc="Testing")
        for inputs, targets in pbar:
            inputs = inputs.squeeze().to('cpu').numpy()

            inputs = feature_extractor(inputs, sampling_rate=16000, padding="max_length",
                                       return_tensors="pt").input_values

            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs).logits

            _, predicted = torch.softmax(outputs, dim=1).max(1)
            correct += torch.eq(predicted, targets).sum().item()

            loss += criterion(outputs, targets).item()
            total += targets.size(0)

    return (correct / total) * 100, loss / len(loader)


def get_outputs(loader, model, feature_extractor, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        pbar = tqdm.tqdm(loader, desc="Testing")
        for inputs, targets in pbar:
            inputs = inputs.squeeze().to('cpu').numpy()

            inputs = feature_extractor(inputs, sampling_rate=16000, padding="max_length",
                                       return_tensors="pt").input_values

            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs).logits

            _, predicted = torch.softmax(outputs, dim=1).max(1)
            all_preds.append(predicted)
            all_targets.append(targets)
        pbar.close()

    return torch.cat(all_preds).cpu(), torch.cat(all_targets).cpu()


# noinspection DuplicatedCode
def test(metadata_path, data_dir, results_dir, model_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model {model_path}")

    # Load the pre-trained model
    model = ASTForAudioClassification.from_pretrained(model_path).to(device)
    # Load the feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Load the dataset
    dataloader = get_dataloader(metadata_path, data_dir, batch_size=batch_size)

    # Calculate test metrics
    preds, targets = get_outputs(dataloader, model, feature_extractor, device)

    f1_macro = f1_score(targets, preds, average="macro")
    f1_micro = f1_score(targets, preds, average="micro")
    f1_weighted = f1_score(targets, preds, average="weighted")
    acc = accuracy_score(targets, preds) * 100
    pre = precision_score(targets, preds, average="macro")
    rec = recall_score(targets, preds, average="macro")
    cm = confusion_matrix(targets, preds)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Confusion Matrix for AST Model")

    disp = ConfusionMatrixDisplay(cm, display_labels=get_classes(metadata_path))

    disp.plot(xticks_rotation='vertical', colorbar=False, values_format='d', ax=ax)
    disp.figure_.savefig(os.path.join(results_dir, "confusion_matrix_ast.png"), bbox_inches="tight")
    print(f"Confusion Matrix Saved!")

    print(f"Accuracy: {acc:.2f}%")
    print(f"F1 Score (macro): {f1_macro:.2f}")
    print(f"F1 Score (micro): {f1_micro:.2f}")
    print(f"F1 Score (weighted): {f1_weighted:.2f}")
    print(f"Precision: {pre:.2f}")
    print(f"Recall: {rec:.2f}")

    output_dict = classification_report(targets, preds, output_dict=True, zero_division=0,
                                        target_names=get_classes(metadata_path))
    df = pd.DataFrame(output_dict).transpose()
    df.to_csv(os.path.join(results_dir, "classification_report_ast.csv"), index=True)
    print(f"Classification Report Saved!")


def time_test(metadata_path, data_dir, model_path, batch_size=64, n_iters=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model {model_path}")
    print(f"Running {n_iters} iterations")

    # Load the pre-trained model
    model = ASTForAudioClassification.from_pretrained(model_path).to(device)
    # Load the feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Load the dataset
    dataloader = get_dataloader(metadata_path, data_dir, batch_size=batch_size)

    avg_time = 0
    std_dev = 0

    for _ in range(n_iters):
        start = time.time()
        _ = get_outputs(dataloader, model, feature_extractor, device)
        end = time.time()

        avg_time += end - start
        std_dev += (end - start) ** 2

    avg_time /= n_iters
    std_dev = (std_dev / n_iters - avg_time ** 2) ** 0.5

    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Standard deviation: {std_dev:.4f}")

    # save the results to txt file
    with open("time_test_results_ast.txt", "w") as f:
        f.write(f"Average time: {avg_time:.4f} seconds\n")
        f.write(f"Standard deviation: {std_dev:.4f} seconds\n")


if __name__ == "__main__":
    main()
