import math
import numpy as np
import pandas as pd
from datasets import load_dataset

import ttest_adapter


def get_model_pair_scores(model_a_b_battles, model_a, model_b):
    score_list = []

    for index, row in model_a_b_battles.iterrows():
        winner = row["winner"]
        score_dict = {row["model_a"]: 0, row["model_b"]: 0}
        if winner == "model_a":
            score_dict[row["model_a"]] = 1
            score_dict[row["model_b"]] = -1
        elif winner == "model_b":
            score_dict[row["model_a"]] = -1
            score_dict[row["model_b"]] = 1

        score_list.append(score_dict)

    scores_df = pd.DataFrame(score_list, columns=[model_a, model_b])

    return scores_df


def pairwise_evaluation(model_a, model_b, battles):
    print(f"Comparing {model_a} and {model_b}")
    # for each pair, we check the current sample size

    model_a_battles = np.logical_or(battles["model_a"] == model_a, battles["model_b"] == model_a)
    model_b_battles = np.logical_or(battles["model_a"] == model_b, battles["model_b"] == model_b)
    model_a_b_battles_filter = np.logical_and(model_a_battles, model_b_battles)

    model_a_b_battles = battles[model_a_b_battles_filter]

    sample_size = len(model_a_b_battles)
    print(f"\tSample size: {sample_size}")

    model_pair_score_df = get_model_pair_scores(model_a_b_battles, model_a, model_b)

    model_a_win_rate = model_pair_score_df[model_pair_score_df[model_a] == 1].shape[0] / sample_size
    model_b_win_rate = model_pair_score_df[model_pair_score_df[model_b] == 1].shape[0] / sample_size
    tie_rate = model_pair_score_df[model_pair_score_df[model_a] == 0].shape[0] / sample_size

    print(f"\t{model_a} win rate: {model_a_win_rate:.2f}")
    print(f"\t{model_b} win rate: {model_b_win_rate:.2f}")
    print(f"\tTie rate: {tie_rate:.2f}")

    t, p, effect_size = ttest_adapter.perform_ttest(model_pair_score_df[model_a], model_pair_score_df[model_b])

    print(f"\tT-test: t={t:.2f}, p={p:.4f}, empirical effect size={effect_size:.2f}")


def main():
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset = load_dataset("lmsys/chatbot_arena_conversations", revision="main")

    battles = dataset["train"].to_pandas()
    battles_no_ties = battles[~battles["winner"].str.contains("tie")]

    # Cohen's d effect size
    effect_size_dict = {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8,
    }

    print("Required sample size for different effect sizes:")
    for effect_name, effect_size in effect_size_dict.items():
        required_sample_size = ttest_adapter.calculate_ttest_sample_size(effect_size)

        required_sample_size = math.ceil(required_sample_size)
        print(f"\teffect size of {effect_size}: {required_sample_size}")

    model_comparison_pair_list = [
        ("gpt-4", "claude-v1"),
        ("gpt-4", "claude-instant-v1"),
        ("gpt-4", "gpt-3.5-turbo"),
    ]

    print("Battles with ties:")
    for model_a, model_b in model_comparison_pair_list:
        pairwise_evaluation(model_a, model_b, battles)

    print("-------------------")
    print("Battles without ties:")
    for model_a, model_b in model_comparison_pair_list:
        pairwise_evaluation(model_a, model_b, battles_no_ties)


if __name__ == "__main__":
    main()
