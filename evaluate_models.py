from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random_forest
import kNN
import gradient_boosting
import desicion_tree
import naiive_bayes
import logistic_regression
import support_vector_machine

# Random Forest Classifier
r = random_forest
accuracy = accuracy_score(r.y_test, r.y_pred)
precision = precision_score(r.y_test, r.y_pred)
recall = recall_score(r.y_test, r.y_pred)
f1 = f1_score(r.y_test, r.y_pred)
print('Accuracy of random forest tree: ', accuracy)
print('Precision of random forest tree: ', precision)
print('Recall of random forest tree: ', recall)
print('F1 score of random forest tree: ', f1)

print('\n')

# K-Nearest Neighbor (kNN)
k = kNN
accuracy = accuracy_score(k.y_test, k.y_pred)
precision = precision_score(k.y_test, k.y_pred)
recall = recall_score(k.y_test, k.y_pred)
f1 = f1_score(k.y_test, k.y_pred)
print('Accuracy of kNN: ', accuracy)
print('Precision of kNN: ', precision)
print('Recall of kNN: ', recall)
print('F1 score of kNN: ', f1)

print('\n')

# Gradient Boosting Classifier
g = gradient_boosting
accuracy = accuracy_score(g.y_test, g.y_pred)
precision = precision_score(g.y_test, g.y_pred)
recall = recall_score(g.y_test, g.y_pred)
f1 = f1_score(g.y_test, g.y_pred)
print('Accuracy of gradient boosting: ', accuracy)
print('Precision of gradient boosting: ', precision)
print('Recall of gradient boosting: ', recall)
print('F1 score of gradient boosting: ', f1)

print('\n')

# Decision Tree
d = desicion_tree
accuracy = accuracy_score(d.y_test, d.y_pred)
precision = precision_score(d.y_test, d.y_pred)
recall = recall_score(d.y_test, d.y_pred)
f1 = f1_score(d.y_test, d.y_pred)
print('Accuracy of decision tree: ', accuracy)
print('Precision of decision tree: ', precision)
print('Recall of decision tree: ', recall)
print('F1 score of decision tree: ', f1)

print('\n')

# Naive Bayes
nb = naiive_bayes
accuracy = accuracy_score(nb.y_test, nb.y_pred)
precision = precision_score(nb.y_test, nb.y_pred)
recall = recall_score(nb.y_test, nb.y_pred)
f1 = f1_score(nb.y_test, nb.y_pred)
print('Accuracy of naiive bayes: ', accuracy)
print('Precision of naiive bayes: ', precision)
print('Recall of naiive bayes: ', recall)
print('F1 score of naiive bayes: ', f1)

print('\n')

# Logistic Regression
lr = logistic_regression
accuracy = accuracy_score(lr.y_test, lr.y_pred)
precision = precision_score(lr.y_test, lr.y_pred)
recall = recall_score(lr.y_test, lr.y_pred)
f1 = f1_score(lr.y_test, lr.y_pred)
print('Accuracy of logistic_regression: ', accuracy)
print('Precision of logistic_regression: ', precision)
print('Recall of logistic_regression: ', recall)
print('F1 score of logistic_regression: ', f1)

print('\n')

# Support Vector Machine
svm = support_vector_machine
accuracy = accuracy_score(svm.y_test, svm.y_pred)
precision = precision_score(svm.y_test, svm.y_pred)
recall = recall_score(svm.y_test, svm.y_pred)
f1 = f1_score(svm.y_test, svm.y_pred)
print('Accuracy of support_vector_machine: ', accuracy)
print('Precision of support_vector_machine: ', precision)
print('Recall of support_vector_machine: ', recall)
print('F1 score of support_vector_machine: ', f1)
