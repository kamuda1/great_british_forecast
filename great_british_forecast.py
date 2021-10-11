from typing import List

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

train_log_odds = log_odds_from_probs(train_probs_win)
test_log_odds = log_odds_from_probs(test_probs_win)


# plt.figure()
# plt.title('Evidence For All Events')
# plt.xlabel('Evidence (logits)')
# plt.ylabel('Freq')
# plt.hist(test_log_odds, 30, density=True, stacked=True, label='test',
#          alpha=1.0)
# plt.hist(train_log_odds, 30, density=True, stacked=True, label='train',
#          alpha=0.6)
# plt.legend()




# Function to generate all binary strings
def generateAllBinaryStrings(n, arr, i):
    """Taken from https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/"""
    if i == n:
        return arr

    # First assign "0" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)

    # And then assign "1" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)


# Driver Code
import itertools


def string_to_bits(bit_string):
    return [int(bit) for bit in bit_string]


def get_all_combos(num_bits: int) -> List:
    all_string_combos = ["".join(seq) for seq in itertools.product("01", repeat=num_bits)]
    all_combos = [string_to_bits(combo) for combo in all_string_combos]
    return all_combos


def get_combos_with_at_least_n_pos(num_bits, n_pos) -> List:
    if n_pos > num_bits:
        n_pos = num_bits

    all_combos = get_all_combos(num_bits)
    all_combos_at_least_n_pos = [combo for combo in all_combos if np.sum(combo) >= n_pos]
    return all_combos_at_least_n_pos


combos_of_concern = get_combos_with_at_least_n_pos(num_bits=13, n_pos=1)
combos_of_concern_npy = np.array(combos_of_concern)
combos_of_concern_probs_win = rf_clf.predict_proba(combos_of_concern_npy)[:, 1]
combos_of_concern_log_odds = log_odds_from_probs(combos_of_concern_probs_win)

# plt.figure()
# plt.hist(combos_of_concern_log_odds, 35)
# plt.show()

combos_of_concern_2 = [combo for combo, log_odds in zip(combos_of_concern, combos_of_concern_log_odds) if log_odds > 0]

where_rules_appear_in_data = []
for combo in combos_of_concern_2:
    tmp_test = []
    for message in all_features:
        pass
        message_binary = [1 if item >= 1 else 0 for item in message]
        tmp_test.append(list(np.bitwise_and(message_binary, combo)) == combo)
    where_rules_appear_in_data.append(tmp_test)

where_rules_appear_in_data = np.array(where_rules_appear_in_data)
plt.figure()
plt.imshow(where_rules_appear_in_data)

plt.figure()
plt.title('How Often Rule in Data')
plt.plot(np.sum(where_rules_appear_in_data, axis=1))

from mpl_toolkits.axes_grid1 import make_axes_locatable

combos_of_concern_2 = [combo for combo, log_odds in zip(combos_of_concern, combos_of_concern_log_odds) if log_odds > 0]

where_rules_appear_in_data = []
for combo in combos_of_concern_2:
    tmp_test = []
    for message in all_features:
        pass
        message_binary = [1 if item >= 1 else 0 for item in message]
        tmp_test.append(list(np.bitwise_and(message_binary, combo)) == combo)
    where_rules_appear_in_data.append(tmp_test)

where_rules_appear_in_data = np.array(where_rules_appear_in_data)
# plt.figure()
# plt.imshow(where_rules_appear_in_data)

# plt.figure()
# plt.title('How Often Rule in Data')
# plt.plot(np.sum(where_rules_appear_in_data, axis=1))

feature_combo_correlation = np.zeros((where_rules_appear_in_data.shape[0], where_rules_appear_in_data.shape[0]))

def correlate_combo_features(feat_1, feat_2):
    """
    Finds the multiplicative correlation of two feature vector combos accross all data. Each feat is a binary array
    of shape (num_episodes, ) corresponding to where the feature (rule combo) occurs in the episodes.
    """
    return np.sum(np.bitwise_and(feat_1, feat_2))

for i in range(feature_combo_correlation.shape[0]):
    for j in range(feature_combo_correlation.shape[0]):
        feature_combo_correlation[i, j] = correlate_combo_features(where_rules_appear_in_data[i],
                                                                   where_rules_appear_in_data[j])

# Infrequent feature combos are unusual and potentially ineffective if they appear to infrequently.
# A combo is infrequent if it's value on the diag is low.

fig, ax = plt.subplots(figsize=(5.5, 5.5))

# the scatter plot:
ax.set_title('Feature Combo Correlation')
ax.imshow(feature_combo_correlation)
ax.set_xlabel('Feature Combo 1')
ax.set_ylabel('Feature Combo 2')
# ax.colorbar()

# Set aspect of the main axes.
ax.set_aspect(1.)

# create new axes on the right and on the top of the current axes
divider = make_axes_locatable(ax)
# below height and pad are in inches
ax_y = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

# make some labels invisible
ax_y.yaxis.set_tick_params(labelleft=False)

combos_of_concern_2_probs_win = rf_clf.predict_proba(combos_of_concern_2)[:, 1]
combos_of_concern_2_log_odds = log_odds_from_probs(combos_of_concern_2_probs_win)
ax_y.barh(list(range(len(combos_of_concern_2_log_odds))), combos_of_concern_2_log_odds)

# plt.figure()
# plt.title('Feature Combo Correlation')
# plt.imshow(feature_combo_correlation)
# plt.xlabel('Feature Combo 1')
# plt.ylabel('Feature Combo 2')
# plt.colorbar()









# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# combos_of_concern_2 = [combo for combo, log_odds in zip(combos_of_concern, combos_of_concern_log_odds) if log_odds > 0]
#
# where_rules_appear_in_data = []
# for combo in combos_of_concern_2:
#     tmp_test = []
#     for message in all_features:
#         pass
#         message_binary = [1 if item >= 1 else 0 for item in message]
#         tmp_test.append(list(np.bitwise_and(message_binary, combo)) == combo)
#     where_rules_appear_in_data.append(tmp_test)
#
# where_rules_appear_in_data = np.array(where_rules_appear_in_data)
# # plt.figure()
# # plt.imshow(where_rules_appear_in_data)
#
# # plt.figure()
# # plt.title('How Often Rule in Data')
# # plt.plot(np.sum(where_rules_appear_in_data, axis=1))
#
#
# def correlate_combo_features(feat_1, feat_2):
#     """
#     Finds the multiplicative correlation of two feature vector combos accross all data. Each feat is a binary array
#     of shape (num_episodes, ) corresponding to where the feature (rule combo) occurs in the episodes.
#     """
#     return np.sum(np.bitwise_and(feat_1, feat_2))
#
# feature_combo_correlation = np.zeros((where_rules_appear_in_data.shape[0], where_rules_appear_in_data.shape[0]))
# for i in range(feature_combo_correlation.shape[0]):
#     for j in range(feature_combo_correlation.shape[0]):
#         feature_combo_correlation[i, j] = correlate_combo_features(where_rules_appear_in_data[i],
#                                                                    where_rules_appear_in_data[j])
#
# # Infrequent feature combos are unusual and potentially ineffective if they appear to infrequently.
# # A combo is infrequent if it's value on the diag is low.
#
# fig, ax = plt.subplots(figsize=(5.5, 5.5))
#
# # the scatter plot:
# ax.set_title('Feature Combo Correlation')
# ax.imshow(np.sum(feature_combo_correlation, axis=1, keepdims=True))
# ax.set_xlabel('Feature Combo 1')
# ax.set_ylabel('Feature Combo 2')
# # ax.colorbar()
#
# # Set aspect of the main axes.
# ax.set_aspect(1.)
#
# # create new axes on the right and on the top of the current axes
# divider = make_axes_locatable(ax)
# # below height and pad are in inches
# ax_y = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
#
# # make some labels invisible
# ax_y.yaxis.set_tick_params(labelleft=False)
# ax_y.set_xlabel('Evidence (Logits)')
#
# combos_of_concern_2_probs_win = rf_clf.predict_proba(combos_of_concern_2)[:, 1]
# combos_of_concern_2_log_odds = log_odds_from_probs(combos_of_concern_2_probs_win)
# ax_y.xaxis.set_label_position("bottom")
# ax_y.barh(list(range(len(combos_of_concern_2_log_odds))), combos_of_concern_2_log_odds)

# plt.figure()
# plt.title('Feature Combo Correlation')
# plt.imshow(feature_combo_correlation)
# plt.xlabel('Feature Combo 1')
# plt.ylabel('Feature Combo 2')
# plt.colorbar()

# plot diag divided by non-diag for each row

# freq_coeff: diag divided by off-diag for each row

freq_coeff = [
    feature_combo_correlation[i, i] /
    np.sum(feature_combo_correlation[i, np.arange(len(feature_combo_correlation)) != i])
    for i in range(len(feature_combo_correlation))]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
# the scatter plot:

freq_coeff = np.nan_to_num(np.array(freq_coeff), copy=False)
print(freq_coeff)
ax.barh(list(range(len(freq_coeff))), freq_coeff * 100)
ax.set_xlabel('Freq Coeff Score (%)')
ax.set_ylabel('Feature Combo')
# ax.colorbar()

# Set aspect of the main axes.
ax.set_aspect(1.)

# create new axes on the right and on the top of the current axes
divider = make_axes_locatable(ax)
# below height and pad are in inches
ax_y = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

# make some labels invisible
ax_y.yaxis.set_tick_params(labelleft=False)
ax_y.set_xlabel('Evidence (Logits)')

combos_of_concern_2_probs_win = rf_clf.predict_proba(combos_of_concern_2)[:, 1]
combos_of_concern_2_log_odds = log_odds_from_probs(combos_of_concern_2_probs_win)
ax_y.xaxis.set_label_position("bottom")
ax_y.barh(list(range(len(combos_of_concern_2_log_odds))), combos_of_concern_2_log_odds)

ax_y.set_title('Frequency Coeff and Evidence for each rule')
ax_y2 = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

# Combine Freq Coeff Score and Logits
ax_y2.yaxis.set_tick_params(labelleft=False)
ax_y2.set_xlabel('Freq Coeff \nScaled (Logits)')

logits_freq_coeff_scaled = combos_of_concern_2_log_odds * freq_coeff

combos_of_concern_2_good = [combo if coeff > 0.2 else None
                            for combo, coeff in zip(combos_of_concern_2, logits_freq_coeff_scaled)]

ax_y2.xaxis.set_label_position("bottom")
ax_y2.barh(list(range(len(logits_freq_coeff_scaled))), logits_freq_coeff_scaled)
ax_y2.yaxis.tick_right()
ax_y2.yaxis.set_label_position('right')
ax_y2.set_yticks(range(len(combos_of_concern_2_good)))
ax_y2.set_yticklabels(combos_of_concern_2_good, size=1)

informative_combos = [combo for combo, coeff in zip(combos_of_concern_2, logits_freq_coeff_scaled) if coeff > 0.2 ]
for combo in informative_combos:
    tmp_combo_prob = rf_clf.predict_proba([combo])[:, 1]
    tmp_combo_log_odds = log_odds_from_probs(tmp_combo_prob)
    print(combo, tmp_combo_log_odds)
