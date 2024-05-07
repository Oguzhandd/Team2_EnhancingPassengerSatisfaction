from sklearn.neighbors import KNeighborsClassifier
from forward_selection import forward_feature_selection

knn = forward_feature_selection(KNeighborsClassifier(n_neighbors=100))
