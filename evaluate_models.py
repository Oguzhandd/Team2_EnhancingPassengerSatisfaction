from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random_forest
import kNN
import gradient_boosting
import desicion_tree

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

print('\n')
# K-Nearest Neighbor (kNN)
k = kNN
accuracy = accuracy_score(k.y_train, k.y_pred)
precision = precision_score(k.y_train, k.y_pred)
recall = recall_score(k.y_train, k.y_pred)
f1 = f1_score(k.y_train, k.y_pred)

print('Accuracy of random forest tree: ', accuracy)
print('Precision of random forest tree: ', precision)
print('Recall of random forest tree: ', recall)
print('F1 score of random forest tree: ', f1)

print('\n')
# Gradient Boosting Classifier
g = gradient_boosting
accuracy = accuracy_score(g.y_train, g.y_pred)
precision = precision_score(g.y_train, g.y_pred)
recall = recall_score(g.y_train, g.y_pred)
f1 = f1_score(g.y_train, g.y_pred)

print('Accuracy of random forest tree: ', accuracy)
print('Precision of random forest tree: ', precision)
print('Recall of random forest tree: ', recall)
print('F1 score of random forest tree: ', f1)

print('\n')
# Decision Tree
d = desicion_tree
accuracy = accuracy_score(d.y_train, d.y_pred)
precision = precision_score(d.y_train, d.y_pred)
recall = recall_score(d.y_train, d.y_pred)
f1 = f1_score(d.y_train, d.y_pred)

print('Accuracy of random forest tree: ', accuracy)
print('Precision of random forest tree: ', precision)
print('Recall of random forest tree: ', recall)
print('F1 score of random forest tree: ', f1)
