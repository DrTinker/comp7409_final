import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

def DrawPlot(data_dict, feature_x, feature_y):
    """
    Draws a plot of the two features selected and colors the POIs.
    feature_list must be of the form ["poi", feature_y, feature_x]
     (i.e. label comes first, then y variable, then x variable (see FeatureFormat write-up)
    """

    feature_list = [feature_x, feature_y, 'poi']
    data_array = featureFormat(data_dict, feature_list)
    # label, features = targetFeatureSplit(data_array)
    poi_color = "r"
    non_poi_color = "g"

    for point in data_array:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            plt.scatter(x, y, color="r")
        else:
            plt.scatter(x, y, color="g")
    plt.scatter(x, y, color="r", label="poi")
    plt.scatter(x, y, color="g", label="non-poi")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.show()

def draw_corr_matrix(data_dict, features_list):
    """
    Draws a correlation matrix plot for the given features list
    """
    # features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
    #                  'salary', 'deferred_income', 'long_term_incentive',
    #                  'poi_email_ratio']
    data = featureFormat(data_dict, features_list, sort_keys=True)
    df = pd.DataFrame(data, columns=features_list)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.8)
    sns.heatmap(corr, mask=False, cmap=cmap, center=0, annot=True,fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()