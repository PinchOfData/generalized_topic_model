# Standard Library
import pathlib
import pickle

# Third Party Library
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def matching_topic(
    score_method,
    dist_type,
    specific_iter,
    num_simulations,
):
    """
    function for matching the true matrix and the estimated matrix \
    (by cosine similarity or correlation coefficient)
        1. column of doc-topic matrix
        2. the row of topic-word matrix

    output:

        corres_num_topic_dict
            {num_sim : {true_topic_idx: estimated_topic_name}}
    """
    if score_method not in ["correlation", "cossim", "dot_product"]:
        raise ValueError(
            "Only three options for soring: correlation, cossim, and dot_product."
        )
    if dist_type not in ["doc_topic", "topic_word"]:
        raise ValueError(
            "Only two distributions are supported: \
            doc_topic or topic_word."
        )
    p = pathlib.Path()
    current_dir = p.cwd()

    corres_num_topic_dict = {}

    iter_name = "iter_{}".format(specific_iter)
    true_path = (
        current_dir.joinpath("data", iter_name, "true_df_{}.pickle".format(dist_type))
        .resolve()
        .as_posix()
    )
    with open(true_path, "rb") as f:
        true_df = pickle.load(f)

    for num in range(num_simulations):
        temp_corres_num_topic_dict = {}
        estimated_n_path = (
            current_dir.joinpath(
                "data", iter_name, "df_{}_{}.pickle".format(dist_type, num)
            )
            .resolve()
            .as_posix()
        )
        with open(estimated_n_path, "rb") as f:
            estimated_df = pickle.load(f)
        if dist_type == "doc_topic":
            score_list = []
            for true_col in true_df.columns:
                true_target_col = true_df.loc[:, true_col]
                score_list_per_row = []
                for col in estimated_df.columns:
                    target_col = estimated_df.loc[:, col]
                    if score_method == "correlation":
                        score_list_per_row.append(
                            np.corrcoef(target_col, true_target_col)[0][1]
                        )
                    elif score_method == "cossim":
                        score_list_per_row.append(
                            np.dot(target_col.T, true_target_col)
                            / (
                                np.linalg.norm(target_col)
                                * np.linalg.norm(true_target_col)
                            )
                        )
                    else:
                        score_list_per_row.append(np.dot(target_col, true_target_col))
                score_list.append(score_list_per_row)
        else:
            score_list = []
            for true_idx in true_df.index:
                true_target_idx = true_df.loc[true_idx, :]
                score_list_per_row = []
                for idx in estimated_df.index:
                    target_idx = estimated_df.loc[idx, :]
                    if score_method == "correlation":
                        score_list_per_row.append(
                            np.corrcoef(target_idx, true_target_idx)[0][1]
                        )
                    elif score_method == "cossim":
                        score_list_per_row.append(
                            np.dot(target_idx.T, true_target_idx)
                            / (
                                np.linalg.norm(target_idx)
                                * np.linalg.norm(true_target_idx)
                            )
                        )
                    else:
                        score_list_per_row.append(np.dot(target_idx, true_target_idx))
                score_list.append(score_list_per_row)

        score_matrix = pd.DataFrame(score_list)
        true_topics, estimated_topics = linear_sum_assignment(-score_matrix)

        for true_topic, estimated_topic in zip(true_topics, estimated_topics):
            temp_corres_num_topic_dict["Topic{}".format(true_topic)] = "Topic{}".format(
                estimated_topic
            )

        corres_num_topic_dict[num] = temp_corres_num_topic_dict

    return corres_num_topic_dict


def calculate_score(
    score_type,
    specific_iter,
    num_simulations,
    corres_num_topic_dict,
):
    """
    A function for calculating the scoring
    input
        score_type: "correlation", "euclid", "cossim", "keywords"

        corres_num_topic_dict (output of creating_dict_for_topic_correspondence)
            {num_sim : {estimated_topic_idx: true_topic}}}
    """
    if score_type not in ["correlation", "euclid", "cossim", "keywords"]:
        raise ValueError(
            "Only four options for scoring similarities: \
            correlation, euclid, cossim, and keywords."
        )
    if score_type in ["correlation", "euclid", "cossim"]:
        dist_type = "doc_topic"
    else:
        dist_type = "topic_word"

    def _rearange_estimated_df(
        df,
        corres_num_topic_dict,
        num_sim,
        dist_type,
    ):
        """
        inner function for rearanging an estimated matrix referencing the true matrix
        """

        corres_dict = corres_num_topic_dict[num_sim]
        if dist_type == "doc_topic":
            reanged_df = df.loc[:, corres_dict.values()]
            reanged_df.columns = corres_dict.keys()
        else:
            reanged_df = df.loc[corres_dict.values(), :]
            reanged_df.index = corres_dict.keys()

        return reanged_df

    def _calc_score_from_two_df(score_type, df1, df2):
        """
        df1 should be true matrix
        df2 should be estimated matrix
        """
        if score_type == "correlation":
            res = df1.corrwith(df2, axis=0).to_list()
        elif score_type == "euclid":
            res = []
            for col in df1.columns:
                series_1 = df1.loc[:, col]
                series_2 = df2.loc[:, col]
                res.append(np.linalg.norm(series_1 - series_2))
        elif score_type == "cossim":
            res = []
            for col in df1.columns:
                series_1 = df1.loc[:, col]
                series_2 = df2.loc[:, col]
                res.append(
                    np.dot(series_1.T, series_2)
                    / (np.linalg.norm(series_1) * np.linalg.norm(series_2))
                )
        else:
            # top 10 keywords approach
            res = []
            df1_topic_keywords_dict, df2_topic_keywords_dict = {}, {}
            for index in df1.index:
                words_1 = df1.loc[index, :]
                df1_topic_keywords_dict[index] = list(
                    words_1.sort_values(ascending=False).index
                )[:10]
                words_2 = df2.loc[index, :]
                df2_topic_keywords_dict[index] = list(
                    words_2.sort_values(ascending=False).index
                )[:10]
            for t_topic in df1_topic_keywords_dict.keys():
                true_words = df1_topic_keywords_dict[t_topic]
                words = df2_topic_keywords_dict[t_topic]
                counter = 0
                for word in words:
                    if word in true_words:
                        counter += 1
                res.append(counter / 10)

        return res

    p = pathlib.Path()
    current_dir = p.cwd()
    iter_name = "iter_{}".format(specific_iter)

    true_df_dist_name = "true_df_{}.pickle".format(dist_type)
    true_df_dist_path = (
        current_dir.joinpath("data", iter_name, true_df_dist_name).resolve().as_posix()
    )
    with open(true_df_dist_path, "rb") as f:
        true_df_dist = pickle.load(f)

    score_list = []
    for num_sim in range(num_simulations):
        target_df_dist_path = (
            current_dir.joinpath(
                "data", iter_name, "df_{}_{}.pickle".format(dist_type, num_sim)
            )
            .resolve()
            .as_posix()
        )
        with open(target_df_dist_path, "rb") as f:
            estimated_df_dist = pickle.load(f)

        reanged_df = _rearange_estimated_df(
            df=estimated_df_dist,
            corres_num_topic_dict=corres_num_topic_dict,
            num_sim=num_sim,
            dist_type=dist_type,
        )
        res = _calc_score_from_two_df(
            score_type=score_type, df1=true_df_dist, df2=reanged_df
        )
        score_list.append(res)

    if dist_type == "doc_topic":
        df_score = pd.DataFrame(score_list, columns=true_df_dist.columns)
    else:
        df_score = pd.DataFrame(score_list, columns=true_df_dist.index)

    return df_score


# def creating_dict_for_topic_correspondence(
#     match_method, dist_type, num_iters, num_simulations, num_topics
# ):
#     """
#     function for checking the topic correspondence \
#     between the true matrix and the estimated matrix \
#     (by cosine similarity or correlation coefficient)
#         1. column of doc-topic matrix
#         2. the row of topic-word matrix

#     Check if the matching from 1. and 2. are the same

#     If it is impossible to match completely, \
#     we exclude it from measuing the simulation performance

#     output:
#         valid_simulation_dict
#             {iter: [which simulation is complete enough to simulate]}
#         corres_num_topic_dict
#             {iter : {num_sim : {true_topic_idx: estimated_topic_name}}}
#     """
#     if match_method not in ["correlation", "cossim"]:
#         raise ValueError("Only two options for matching:correlation, cossim.")
#     if dist_type not in ["doc_topic", "topic_word"]:
#         raise ValueError(
#             "Only two distributions are supported: \
#             doc_topic or topic_word."
#         )

#     p = pathlib.Path()
#     current_dir = p.cwd()
#     valid_simulation_dict = {}
#     corres_num_topic_dict = {}

#     for n_iter in range(num_iters):
#         sub_corres_num_topic_dict = {}
#         iter_name = "iter_{}".format(n_iter)
#         true_path = (
#             current_dir.joinpath(
#                 "data", iter_name, "true_df_{}.pickle".format(dist_type)
#             )
#             .resolve()
#             .as_posix()
#         )
#         with open(true_path, "rb") as f:
#             true_df = pickle.load(f)

#         valid_sim_nums_list = []
#         for num in range(num_simulations):
#             match_key = []
#             temp_corres_num_topic_dict = {}
#             estimated_n_path = (
#                 current_dir.joinpath(
#                     "data", iter_name, "df_{}_{}.pickle".format(dist_type, num)
#                 )
#                 .resolve()
#                 .as_posix()
#             )
#             with open(estimated_n_path, "rb") as f:
#                 df_n = pickle.load(f)

#             if dist_type == "doc_topic":
#                 for true_col in true_df.columns:
#                     true_target_col = true_df.loc[:, true_col]
#                     res = {}
#                     for i, col in enumerate(df_n.columns):
#                         target_col = df_n.loc[:, col]
#                         if match_method == "correlation":
#                             res[i] = np.corrcoef(target_col, true_target_col)[0][1]
#                         else:
#                             res[i] = np.dot(target_col.T, true_target_col) / (
#                                 np.linalg.norm(target_col)
#                                 * np.linalg.norm(true_target_col)
#                             )
#                     key, _ = max(res.items(), key=lambda x: x[1])
#                     match_key.append(key)
#                     temp_corres_num_topic_dict[true_col] = "Topic{}".format(key)
#             else:
#                 for true_idx in true_df.index:
#                     true_target_idx = true_df.loc[true_idx, :]
#                     res = {}
#                     for i, idx in enumerate(df_n.index):
#                         target_idx = df_n.loc[idx, :]
#                         if match_method == "correlation":
#                             res[i] = np.corrcoef(target_idx, true_target_idx)[0][1]
#                         else:
#                             res[i] = np.dot(target_idx.T, true_target_idx) / (
#                                 np.linalg.norm(target_idx)
#                                 * np.linalg.norm(true_target_idx)
#                             )
#                     key, _ = max(res.items(), key=lambda x: x[1])
#                     match_key.append(key)
#                     temp_corres_num_topic_dict[true_idx] = "Topic{}".format(key)

#             if len(set(match_key)) == num_topics:
#                 valid_sim_nums_list.append(num)
#                 sub_corres_num_topic_dict[num] = temp_corres_num_topic_dict

#         valid_simulation_dict[num] = valid_sim_nums_list
#         corres_num_topic_dict[num] = sub_corres_num_topic_dict

#     return valid_simulation_dict, corres_num_topic_dict