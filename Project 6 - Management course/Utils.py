import matplotlib.pyplot as plt


def plot(result, model_name):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    n_train = range(1, len(result["train_loss"]) + 1)
    n_test = range(1, len(result["test_loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    axes[0].scatter(n_train, result["train_loss"], label="train_loss")
    axes[0].scatter(n_test, result["test_loss"], label="test_loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].scatter(n_train, result["train_acc"], label="train_acc")
    axes[1].scatter(n_test, result["test_acc"], label="test_acc")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.savefig(f"result_{model_name}.svg", format="svg")

    plt.tight_layout()
    plt.show()

