import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# Define the list of classes
classes_sinhala_words = ['සැලසේ', 'පෞද්ගලික', 'ක්ෂේත්‍රයන්ට', 'ඇතිවේ', 'වෙන් කෙරේ', 'වැසි', 'විශ්වවිද්‍යාලය', 'මම', 'ගෙදර', 'හා', 'ණය', 'අධ්‍යාපනය', 'රාත්‍රියේ දී', 'පහසුකම්', 'යමි', 'මුදල්', 'ශිෂ්‍යයන්ට', 'ආර්ථික']

# Generate random predicted labels for the classes with 90% accuracy
num_samples = 1000
true_labels = np.random.choice(classes_sinhala_words, num_samples)
predicted_labels = [label if random.random() <= 0.9 else random.choice(classes_sinhala_words) for label in true_labels]

# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=classes_sinhala_words)

cm = np.array(cm)  # Example confusion matrix

# Convert values to percentages
cm_percent = (cm / cm.sum(axis=1)[:, np.newaxis]) * 100





# Define class labels
labels = ['සැලසේ', 'පෞද්ගලික', 'ක්ෂේත්‍රයන්ට', 'ඇතිවේ', 'වෙන් කෙරේ', 'වැසි', 'විශ්වවිද්‍යාලය', 'මම', 'ගෙදර', 'හා', 'ණය', 'අධ්‍යාපනය', 'රාත්‍රියේ දී', 'පහසුකම්', 'යමි', 'මුදල්', 'ශිෂ්‍යයන්ට', 'ආර්ථික']

# Create heatmap
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(cm_percent, cmap='Blues')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Percentage', rotation=-90, va='bottom')

# Set tick labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Rotate the tick labels and set alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, f'{cm_percent[i, j]:.1f}%', ha='center', va='center', color='black')

# Set title and labels
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')

# Show the plot
plt.show()