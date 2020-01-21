import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

def preprocess_col_names(df):
    """
    Replace strings in column names for easier preprocessing.

    Parameters
    ----------
    first : pandas DataFrame
        input dataframe for which column names have to be formatted,
        name `df`

    Returns
    -------
    dataframe
        input dataframe with column names changed

    Raises
    ------
    TODO Error handling code
    """
    
    temp_df = df.copy()

    list_str_replace = ["'s ", "/", " ("]
    list_str_remove = [")"]
    list_new_colnames = []
    for col in temp_df.columns.values:

        # replace strings to remove with blank
        for str in list_str_remove:
            if str in col:
                col.replace(str, '')

        # replace other strings with underscore
        for str in list_str_replace:
            if str in col:
                col.replace(str, '_')

        list_new_colnames.append(col)

    temp_df.columns = list_new_colnames
    return temp_df


def visualize_tsne_2d(df, feature_cols, perplexity=25):
    """
    Visuzalize high dimensional data using TSNE in 2 dimensions.

    Parameters
    ----------
    df : pandas DataFrame
        Input dataframe from which TSNE plot is to be created.
    perplexity : integer, typically in the range of 5-50
        Perplexity parameter for TSNE
        smaller the value, more localized the clustering
        for larger values, global patterns are prioritized over local effects.
    feature_cols : list of strings
        List of feature column names from the input dataframe.

    Returns
    -------
    The function doesn't return any data
    The TSNE plot is displayed for input data

    Raises
    ------
    TODO Error handling code
    """
    # instantiate and fit tsne
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity)
    tsne_results = tsne.fit_transform(df.values)

    # store results of tsne in a dataframe
    tsne_subset = pd.DataFrame()
    tsne_subset['tsne-2d-one'] = tsne_results[:,0]
    tsne_subset['tsne-2d-two'] = tsne_results[:,1]

    # plot tsne in 2d space
    fig = plt.figure(figsize=(16, 10))
    ax = fig.gca()
    sns.scatterplot(
                    x="tsne-2d-one", y="tsne-2d-two",
                    data=tsne_subset,
                    alpha=0.3, ax=ax
                   )
    ax.set_title('TSNE for perplexity = '+str(perplexity))
    ax.set_xlabel('TSNE dimension 1')
    ax.set_ylabel('TSNE dimension 2')


def drop_features_vif(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    return X.iloc[:, variables]


# TODO PCA code if required
# pca_all = PCA()
# pca_all.fit(train_array)
# pca_ev_ratio = pca_all.explained_variance_ratio_
# sns.barplot(x=np.arange(1,len(pca_ev_ratio)+1),y=pca_ev_ratio)
# dplt.xlabel('# Components')
# plt.ylabel('Explained Variance Ratio')
# plt.tight_layout()
# sns.lineplot(x=np.arange(1,len(pca_ev_ratio)+1),y=np.cumsum(pca_ev_ratio))
# plt.xlabel('# Principal Components')
# plt.ylabel('Fraction of cumulative \n explained variance')
# plt.tight_layout()