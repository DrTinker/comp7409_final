import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.feature_selection import SelectKBest
from pprint import pprint

# with open("final_project_dataset.pkl", "rb") as data_file:
#     data_dict = pickle.load(data_file)
#
# features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus',
#                       'salary', 'deferred_income', 'long_term_incentive',
#                       'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
#                       'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']
# k=8
def Select_K_Best(data_dict, features_list, k):
    """
    Runs scikit-learn's SelectKBest feature selection algorithm, returns an
    array of tuples with the feature and its score.
    """

    data_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data_array)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    tuples = zip(features_list[1:], scores)
    k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)

    return k_best_features[:k]
