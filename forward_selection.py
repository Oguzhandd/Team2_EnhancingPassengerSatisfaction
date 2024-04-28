from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector as SFS


sfs = SFS(
    estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    n_features_to_select='auto',
    tol=0.001,
    # chosen direction in the article
    direction='forward',
    # chosen scoring in the article
    scoring='roc_auc',
    # cv = 10 = 10-fold cross-validation
    cv=10,
)
