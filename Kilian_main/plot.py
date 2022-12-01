from matplotlib import pyplot as plt


def plot_loss(losses, val_losses, file_name, epoch):
    loss = losses
    val_loss = val_losses
    epochs = range(1, len(loss) + 1)
    plt.figure(epoch+100)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('val_Loss')
    plt.legend()
    plt.savefig(file_name)
    plt.close

def plot_accuracy(acc, val_acc, file_name, epoch):
    accuracy = acc
    val_accuracy = val_acc
    epochs = range(1, len(accuracy) + 1)
    plt.figure(epoch)
    plt.plot(epochs, accuracy, 'y', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(file_name)
    plt.close


