from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random_forest

# Random Forest Classifier
r = random_forest
accuracy = accuracy_score(r.y_train, r.y_pred)
precision = precision_score(r.y_train, r.y_pred)
recall = recall_score(r.y_train, r.y_pred)
f1 = f1_score(r.y_train, r.y_pred)

print('Accuracy of random forest tree: ', accuracy)
print('Precision of random forest tree: ', precision)
print('Recall of random forest tree: ', recall)
print('F1 score of random forest tree: ', f1)

# K-Nearest Neighbor (kNN)

