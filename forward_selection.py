from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold


def forward_feature_selection(classifier):
    # stratified sampling of 10-fold cross-validation
    sfK = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    sfs = SFS(
        estimator=classifier,
        n_features_to_select='auto',
        tol=0.001,
        # chosen direction in the article
        direction='forward',
        # chosen scoring in the article
        scoring='roc_auc',
        # cv = 10 = 10-fold cross-validation
        cv=sfK,
    )
    return sfs
