from sklearn.feature_selection import SequentialFeatureSelector as SFS


def forward_feature_selection(classifier):
    sfs = SFS(
        estimator=classifier,
        n_features_to_select='auto',
        tol=0.001,
        # chosen direction in the article
        direction='forward',
        # chosen scoring in the article
        scoring='roc_auc',
        # cv = 10 = 10-fold cross-validation
        cv=10,
    )
    return sfs
