from sklearn.ensemble import GradientBoostingClassifier
from forward_selection import forward_feature_selection

gb_model = forward_feature_selection(GradientBoostingClassifier(n_estimators=100))
