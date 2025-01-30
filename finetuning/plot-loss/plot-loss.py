# Code to plot the loss after training the model (adapted form chatGPT)

import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def plot_loss_average(log_file, output_dir, max_epochs=3):
    # Load logs
    with open(log_file, 'r') as f:
        logs = json.load(f)

    # Ensure log_history exists
    log_history = logs.get("log_history", [])
    if not log_history:
        print("No 'log_history' found in the JSON file.")
        return

    # Extract training and evaluation loss
    train_loss_per_epoch = defaultdict(list)  #Store training losses by epoch
    eval_loss = []
    eval_epochs = []

    for entry in log_history:
        if "loss" in entry and entry["epoch"] <= max_epochs:  #training loss
            epoch = int(entry["epoch"])
            train_loss_per_epoch[epoch].append(entry["loss"])
        if "eval_loss" in entry and entry["epoch"] <= max_epochs:  #evaluatoin loss
            eval_loss.append(entry["eval_loss"])
            eval_epochs.append(entry["epoch"])

    # Compute the average training loss per epoch
    train_loss = [
        sum(losses) / len(losses) for epoch, losses in sorted(train_loss_per_epoch.items())
    ]
    epochs = list(sorted(train_loss_per_epoch.keys()))

    # Plot training and evaluation loss
    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(epochs, train_loss, label="Training Loss", marker='o')
    if eval_loss:
        plt.plot(eval_epochs, eval_loss, label="Evaluation Loss", marker='o', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "final_loss.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    plt.show()

log_file = "/home/vellard/playlist_continuation/finetuning/fine_tuned_model/checkpoint-295830/trainer_state.json"
output_dir = "/home/vellard/playlist_continuation/finetuning/plot-loss"

plot_loss_average(log_file, output_dir, max_epochs=30)
