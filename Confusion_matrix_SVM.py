import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example values (replace with real values)
y_true = [0,0,1,1,0,1,0,1]
y_pred = [0,1,1,1,0,0,0,1]

cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Alert", "Drowsy"],
    yticklabels=["Alert", "Drowsy"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()
