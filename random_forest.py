from sklearn.ensemble import RandomForestClassifier

from forward_selection import forward_feature_selection

random_forest = forward_feature_selection(RandomForestClassifier(n_estimators=100))
