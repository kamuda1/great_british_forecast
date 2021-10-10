from great_british_utils import *

import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
all_features, all_targets, all_data_df = get_features_and_targets()

# Get features and Train
all_features_npy = np.vstack(all_features)
all_targets_npy = np.array(all_targets)

clf = MultinomialNB()
clf.fit(all_features_npy, all_targets_npy)

# Check out f-score
y_pred = clf.predict(all_features_npy)
print(precision_recall_fscore_support(all_targets_npy, y_pred))

all_data_df['prob_winning'] = all_data_df.apply(lambda row: clf.predict_proba([row['score_freq']]), axis=1)

# Train something cooler with a train/test split
train_features_npy, test_features_npy, train_targets_npy, test_targets_npy = train_test_split(
    all_features_npy, all_targets_npy, test_size=0.33, random_state=42)
rf_clf = RandomForestClassifier()
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

def log_odds_from_probs(probs):
    odds = odds_from_prob(np.array(clean_probs(probs)))
    log_odds = np.log(odds)
    return log_odds

train_log_odds = log_odds_from_probs(train_probs_win)
test_log_odds = log_odds_from_probs(test_probs_win)


plt.figure()
plt.title('Evidence For All Events')
plt.xlabel('Evidence (logits)')
plt.ylabel('Freq')
plt.hist(test_log_odds, 30, density=True, stacked=True, label='test',
         alpha=1.0)
plt.hist(train_log_odds, 30, density=True, stacked=True, label='train',
         alpha=0.6)
plt.legend()
