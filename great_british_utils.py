from typing import List

import requests
import re
import numpy as np
import pandas
import pandas as pd
from bs4 import BeautifulSoup


def etl_seasons_data():
    """
    Loads data from all seasons into a pandas dataframe.
    TODO: Cleanup into smaller functions
    TODO: Add type hints
    """
    all_seasons = []
    final_winners = []

    for season_index in range(1, 12):
        print(season_index)
        response = requests.get('https://en.wikipedia.org/wiki/The_Great_British_Bake_Off_(series_' + str(season_index) + ')')
        soup = BeautifulSoup(response.text, 'html.parser')

        baker_background_section = str(soup.find_all('table')[1])
        baker_background_section_pd = pandas.read_html(baker_background_section, attrs={"class": "wikitable"})[0]

        episode_list = []
        for episode_index in range(10):
            test_section = str(soup.find_all('table')[3 + episode_index])
            test_section_pd = pandas.read_html(test_section, attrs={"class": "wikitable"})[0]
            episode_list.append(test_section_pd)
            test_section = test_section.replace('gold;', 'gold')
            test_section = test_section.replace('\n</td>', '</td>')
            if len(test_section_pd) == 3:
                try:
                    line_with_winner = test_section.lower().split('gold">\n')[1].split('\n')[0]
                    line_with_winner_all_messed_up = line_with_winner.split('<')
                    line_with_winner_all_messed_up = [line.split('>') for line in line_with_winner_all_messed_up]
                    line_with_winner_all_messed_up = [item for sublist in line_with_winner_all_messed_up for item in
                                                      sublist]
                    final_winner = [line for line in line_with_winner_all_messed_up if '/' not in line and len(line) > 0
                                    and line != 'td' and '=' not in line][0]
                    final_winners.append(final_winner)
                except:
                    print(5)
                break
        all_seasons.append(episode_list)
    return all_seasons, final_winners


def return_one_hot_in_n(pos_ind: int, n: int = 12):
    """
    Returns an zeros array of length n where pos_ind is 1
    """
    result = np.zeros([n])
    result[pos_ind] = 1
    return result


def string_to_int(string, max_val):
    if has_numbers(string):
        return int(re.search(r'\d+', string).group())
    else:
        return max_val


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


def remove_baker(df: pd.DataFrame, removed_bakers: List) -> pd.DataFrame:
    """
    Removes a list of bakers from the bakers column.
    """
    for removed_baker in removed_bakers:
        df = df[df['Baker'] != removed_baker]
    return df


def process_episode(bakers_results_season_pd, episode_idx, episode_pd, all_features, all_targets, all_targets_per_episode,
                    all_data_df, orig_num_bakers, season_index, episode_list, temp_winner):
    """
    Processes an episode.
    TODO: break & clean up this monster
    """
    episode_pd.fillna(str(orig_num_bakers) + 'th', inplace=True)

    if len(set(episode_pd['Baker'])) == 3:
        return bakers_results_season_pd, all_features, all_targets, all_targets_per_episode, all_data_df

    removed_bakers = (set(episode_pd['Baker']) - set(episode_list[episode_idx + 1]['Baker']))
    print(removed_bakers)

    try:
        episode_rankings = [return_one_hot_in_n(string_to_int(i, orig_num_bakers) - 1, orig_num_bakers) for i in
                            episode_pd[episode_pd.columns[2]]]
    except:
        print('Oh no!')

    bakers_results_season_pd['rank_freq_tmp'] = episode_rankings
    bakers_results_season_pd['removed'] = False
    for removed_baker in removed_bakers:
        bakers_results_season_pd['removed'] = np.select([bakers_results_season_pd['Baker'] == removed_baker], [True])
    bakers_results_season_pd['rank_freq'] += bakers_results_season_pd['rank_freq_tmp']

    tmp_features = list(bakers_results_season_pd['rank_freq'])

    while len(tmp_features[0]) < 13:
        for i in range(len(tmp_features)):
            tmp_features[i] = np.append(tmp_features[i], 0)

    for baker_idx, baker in enumerate(bakers_results_season_pd['Baker']):
        tmp_data_df = pd.DataFrame.from_dict({
            'baker': baker,
            'season': season_index,
            'episode': episode_idx,
            'score_freq': [tmp_features[baker_idx]]})
        all_data_df = all_data_df.append(tmp_data_df, ignore_index=True)

    all_features += tmp_features
    all_targets_per_episode += list(bakers_results_season_pd['removed'])
    bakers_results_season_pd.reset_index(inplace=True, drop=True)
    temp_winner_idx = bakers_results_season_pd.index[
        bakers_results_season_pd['Baker'].str.lower() == temp_winner].tolist()[0]
    tmp_target = np.zeros(len(tmp_features))
    try:
        tmp_target[temp_winner_idx] = 1
    except:
        print(5)
    all_targets += list(tmp_target)

    print('Save results here')
    bakers_results_season_pd = remove_baker(bakers_results_season_pd, removed_bakers)
    return bakers_results_season_pd, all_features, all_targets, all_targets_per_episode, all_data_df


def get_features_and_targets():
    """ Does all the things. TODO: organize & atomize """
    all_seasons, final_winners = etl_seasons_data()

    # init final data lists and dataframe
    all_features = []
    all_targets = []
    all_targets_per_episode = []
    column_names = ['baker', 'season', 'episode', 'score_freq']
    all_data_df = pd.DataFrame(columns=column_names)

    # process all data
    for season_index, episode_list in enumerate(all_seasons):
        bakers_results_season_pd = pd.DataFrame(episode_list[0]['Baker'])
        orig_num_bakers = bakers_results_season_pd.shape[0]
        bakers_results_season_pd['rank_freq'] = orig_num_bakers * [np.zeros([orig_num_bakers])]

        temp_winner = final_winners[season_index]

        for episode_idx, episode_pd in enumerate(episode_list):
            bakers_results_season_pd, all_features, all_targets, all_targets_per_episode, all_data_df = process_episode(
                bakers_results_season_pd, episode_idx, episode_pd, all_features, all_targets, all_targets_per_episode,
                all_data_df, orig_num_bakers, season_index, episode_list, temp_winner)

    return all_features, all_targets, all_data_df
