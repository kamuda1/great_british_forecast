from great_british_utils import get_features_and_targets

import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
all_features, all_targets, all_data_df = get_features_and_targets()

# Do the ML
all_features_npy = np.vstack(all_features)
all_targets_npy = np.array(all_targets)

clf = MultinomialNB()
clf.fit(all_features_npy, all_targets_npy)

y_pred = clf.predict(all_features_npy)
print(precision_recall_fscore_support(all_targets_npy, y_pred))

all_data_df['prob_winning'] = all_data_df.apply(lambda row: clf.predict_proba([row['score_freq']]), axis=1)

train_features_npy, test_features_npy, train_targets_npy, test_targets_npy = train_test_split(
    all_features_npy, all_targets_npy, test_size=0.33, random_state=42)
rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
rf_clf.fit(train_features_npy, train_targets_npy)

print('Train')
print(precision_recall_fscore_support(train_targets_npy, rf_clf.predict(train_features_npy)))
print('Test')
print(precision_recall_fscore_support(test_targets_npy, rf_clf.predict(test_features_npy)))

all_data_df['prob_winning'] = all_data_df.apply(lambda row: rf_clf.predict_proba([row['score_freq']]), axis=1)
all_data_tmp = all_data_df[all_data_df['baker'] == 'Jo']

# play with probs and evidence
train_probs_win = rf_clf.predict_proba(train_features_npy)[:, 1]
test_probs_win = rf_clf.predict_proba(test_features_npy)[:, 1]


def odds_from_prob(prob):
    return prob / (1 - prob)


train_odds = odds_from_prob(train_probs_win)
test_odds = odds_from_prob(test_probs_win)

train_log_odds = np.log(train_odds)
test_log_odds = np.log(test_odds)

plt.figure()
plt.title('test_log_odds')
plt.xlabel('Log Odds')
plt.ylabel('Freq')
plt.hist(test_log_odds, density=True, stacked=True, label='test',
         alpha=1.0)
plt.hist(train_log_odds, density=True, stacked=True, label='train',
         alpha=0.6)
plt.legend()

print(5)
