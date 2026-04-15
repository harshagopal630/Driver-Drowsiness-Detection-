import matplotlib.pyplot as plt

# Example accuracy values (replace with real)
epochs = [1,2,3,4,5,6,7,8,9,10]

train_acc = [70,75,80,83,86,88,90,92,94,95]
val_acc = [68,73,78,81,84,86,88,90,92,93]

plt.figure()

plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend(["Training Accuracy", "Validation Accuracy"])

plt.savefig("accuracy_graph.png")

plt.show()
