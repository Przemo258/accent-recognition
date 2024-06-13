import os
import tqdm
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from ClassificationNet import ClassificationNet
from torch.utils.tensorboard import SummaryWriter
from MozillaCommonVoiceDatasetCNN import get_classes
from torch.optim.lr_scheduler import CosineAnnealingLR
from MozillaCommonVoiceDatasetCNN import get_dataloader, get_num_classes
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, classification_report


def main():
    train_metadata = r"../../data/metadata/train.csv"
    valid_metadata = r"../../data/metadata/val.csv"
    test_metadata = r"../../data/metadata/test.csv"
    data_dir = r"../../data/clips/"

    model_save_dir = "../../models/cnn/"
    model_to_test = "../../models/cnn/classification_net_20.pt"

    results_dir = "../../data/plots/"

    num_epochs = 20
    batch_size = 10
    learning_rate = 1e-5

    train(train_metadata, valid_metadata, data_dir, num_epochs=num_epochs, batch_size=batch_size,
          learning_rate=learning_rate,
          save_dir=model_save_dir)
    test(test_metadata, data_dir, results_dir, model_to_test)
    time_test(test_metadata, data_dir, model_to_test)


def train(train_metadata, valid_metadata, data_dir, num_epochs=2, batch_size=6, learning_rate=1e-5, save_dir="models/",
          load_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    print(f"Training model for {num_epochs} epochs")

    # Load the datasets
    train_dataloader = get_dataloader(train_metadata, data_dir, augment=True, batch_size=batch_size, device=device)
    valid_dataloader = get_dataloader(valid_metadata, data_dir, batch_size=batch_size, device=device)

    # Load the model
    n_classes = get_num_classes(train_metadata)

    model = ClassificationNet(n_classes=n_classes).to(device)

    # in case we want to continue training from a saved model
    if load_model:
        model.load_state_dict(torch.load(load_model))
        print(f"Loaded model from {load_model}")

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs + 1, eta_min=1e-9)

    # Define the loss function
    criterion = CrossEntropyLoss()

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total = 0

        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.softmax(outputs, dim=1).max(1)
            epoch_correct += torch.eq(predicted, targets).sum().item()
            epoch_loss += loss.item()
            total += len(targets)

            pbar.set_postfix_str(f"Batch Loss: {loss.item():.4f} Epoch Accuracy: {(epoch_correct / total) * 100:.2f}%")
            writer.add_scalar("Loss/train", loss.item(), epoch)

        print(f"Calculating validation metrics...")

        writer.add_scalar("Accuracy/train", epoch_correct / total, epoch)

        # Update the learning rate
        lr_scheduler.step()

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"classification_net_epoch_{epoch + 1}.pt"))

        # Calculate the validation loss and accuracy
        val_acc, val_loss = calculate_metrics(valid_dataloader, model, criterion, device)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # save lr to tensorboard
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        print(f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_acc:.2f}%")

        time.sleep(1)


def calculate_metrics(loader, model, criterion, device):
    model.eval()

    correct = 0
    total = 0
    loss = 0

    with torch.inference_mode():
        pbar = tqdm.tqdm(loader, desc="Testing")

        for inputs, targets in pbar:
            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.softmax(outputs, dim=1).max(1)

            total += targets.size(0)
            correct += torch.eq(predicted, targets).sum().item()

            loss += criterion(outputs, targets).item()

            accuracy = (correct / total) * 100
            total_loss = loss / len(loader)

    return accuracy, total_loss


def get_outputs(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        pbar = tqdm.tqdm(loader, desc="Testing")

        for inputs, targets in pbar:
            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.softmax(outputs, dim=1).max(1)

            all_preds.append(predicted)
            all_targets.append(targets)

    return torch.cat(all_preds).cpu(), torch.cat(all_targets).cpu()


# noinspection DuplicatedCode
def test(metadata_path, data_dir, results_dir, model_path, batch_size=64, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model {model_path}")

    model = ClassificationNet().to(device)
    model.load_state_dict(torch.load(model_path))

    data_loader = get_dataloader(metadata_path, data_dir, batch_size=batch_size, device=device)

    preds, targets = get_outputs(data_loader, model, device)

    f1_macro = f1_score(targets, preds, average="macro")
    f1_micro = f1_score(targets, preds, average="micro")
    f1_weighted = f1_score(targets, preds, average="weighted")
    acc = accuracy_score(targets, preds) * 100
    pre = precision_score(targets, preds, average="macro")
    rec = recall_score(targets, preds, average="macro")
    cm = confusion_matrix(targets, preds)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Confusion Matrix for CNN Model")

    disp = ConfusionMatrixDisplay(cm, display_labels=get_classes(metadata_path))

    disp.plot(xticks_rotation='vertical', colorbar=False, values_format='d', ax=ax)
    disp.figure_.savefig(os.path.join(results_dir, "confusion_matrix_cnn.png"), bbox_inches="tight")
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
    df.to_csv(os.path.join(results_dir, "classification_report_cnn.csv"), index=True)
    print(f"Classification Report Saved!")


def time_test(metadata_path, data_dir, model_path, batch_size=64, n_iters=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model {model_path}")
    print(f"Running {n_iters} iterations")

    model = ClassificationNet().to(device)
    model.load_state_dict(torch.load(model_path))

    data_loader = get_dataloader(metadata_path, data_dir, batch_size=batch_size, device=device)

    avg_time = 0
    std_dev = 0

    for _ in range(n_iters):
        start = time.time()
        get_outputs(data_loader, model, device)
        end = time.time()

        avg_time += end - start
        std_dev += (end - start) ** 2

    avg_time /= n_iters
    std_dev = (std_dev / n_iters - avg_time ** 2) ** 0.5

    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Standard deviation: {std_dev:.4f} seconds")

    # save results to txt file
    with open("time_test_results_cnn.txt", "w") as f:
        f.write(f"Average time: {avg_time:.4f} seconds\n")
        f.write(f"Standard deviation: {std_dev:.4f} seconds\n")


if __name__ == "__main__":
    main()
