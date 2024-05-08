import lzma
import pickle
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


RESULT_DIR = Path("5_04_result")


def safe_literal_eval(s: str) -> Any:
    try:
        # Attempt to evaluate the string as a Python literal
        if s == "[]":
            return []
        return literal_eval(s)
    except ValueError:
        # Return the original string if literal_eval fails
        return s


def beta_ema(beta_list: List[float], factor: float) -> float:
    beta_ema = beta_list[0]

    for i in range(1, len(beta_list)):
        beta_ema = beta_ema * (1 - factor) + beta_list[i] * factor

    return beta_ema


def process_result_file(fname: Path) -> Tuple[str, pd.DataFrame]:
    # read only the first 100 lines
    with open(fname, "r") as f:
        config = f.readline()
    df = pd.read_csv(fname, sep="|", header=None, skiprows=1)
    df.columns = [
        "dummy",
        "accept_cnt",
        "previous_beta_list",
        "draft_probs",
        "accept_cnt_list",
        "accept_probs",
        "current_beta_list",
    ]

    df["accept_cnt"] = df["accept_cnt"].astype(int)
    df["previous_beta_list"] = df["previous_beta_list"].apply(
        lambda x: safe_literal_eval(x.strip())
    )
    df["draft_probs"] = df["draft_probs"].apply(lambda x: safe_literal_eval(x.strip()))
    df["accept_cnt_list"] = df["accept_cnt_list"].apply(
        lambda x: safe_literal_eval(x.strip())
    )
    df["accept_probs"] = df["accept_probs"].apply(
        lambda x: safe_literal_eval(x.strip())
    )
    df["current_beta_list"] = df["current_beta_list"].apply(
        lambda x: safe_literal_eval(x.strip())
    )

    df = df.dropna()
    df = df.drop(columns=["dummy"])

    # if len of previous_beta_list is 0, drop the row
    df = df[df["previous_beta_list"].map(len) > 0]

    # apply beta_ema to previous_beta_list
    df["beta_ema"] = df["previous_beta_list"].apply(lambda x: beta_ema(x, 0.5))

    return config, df


def get_accept_cnt_vs_draft_prob(df: pd.DataFrame) -> pd.DataFrame:
    # if the index is smaller than accept_cnt, then it is accepted
    # if the index is exactly same to accept_cnt, then it is rejected
    # if the index is larger than accept_cnt, then it is not considered

    # Initialize an empty list to store data
    data = []

    # Iterate over each row in the original dataframe
    for _, row in df.iterrows():
        beta_ema = row["beta_ema"]
        # Iterate over each draft_prob in the row
        for i, draft_prob in enumerate(row["draft_probs"]):
            # Check if the current index matches or is less than the accept count
            if i < row["accept_cnt"]:
                accepted = 1
            elif i == row["accept_cnt"]:
                accepted = 0
            else:
                continue

            # Append a tuple to the list
            data.append((beta_ema, draft_prob, accepted))

    # Create the DataFrame from the list
    df_draft = pd.DataFrame(data, columns=["beta_ema", "draft_prob", "accepted"])
    return df_draft


def bin_draft_accepted_df(
    draft_accepted_df: pd.DataFrame, num_bins: int = 20
) -> pd.DataFrame:
    # Define bins
    beta_ema_bins = np.linspace(0, 1, num_bins + 1)
    draft_prob_bins = np.linspace(0, 1, num_bins + 1)

    # Binning the data
    draft_accepted_df["beta_ema_binned"] = pd.cut(
        draft_accepted_df["beta_ema"],
        bins=beta_ema_bins,
        labels=np.linspace(0, 1, num_bins, endpoint=False),
    )
    draft_accepted_df["draft_prob_binned"] = pd.cut(
        draft_accepted_df["draft_prob"],
        bins=draft_prob_bins,
        labels=np.linspace(0, 1, num_bins, endpoint=False),
    )

    # Group and aggregate data
    binned_df = (
        draft_accepted_df.groupby(["beta_ema_binned", "draft_prob_binned"])
        .agg({"accepted": "mean"})
        .fillna(0)
    )
    binned_df.reset_index(inplace=True)

    # Convert categories to codes to be used in regression
    binned_df["beta_ema_binned_code"] = binned_df["beta_ema_binned"].cat.codes
    binned_df["draft_prob_binned_code"] = binned_df["draft_prob_binned"].cat.codes

    return binned_df


def main(args: Namespace):
    fname = Path(args.file)
    config, df = process_result_file(fname)
    draft_accepted_df = get_accept_cnt_vs_draft_prob(df)
    binned_df = bin_draft_accepted_df(draft_accepted_df)
    result_dict = {
        "config": config,
        "result_df": df,
        "draft_accepted_df": draft_accepted_df,
        "binned_df": binned_df,
    }
    with lzma.open(fname.with_suffix(".xz"), "wb") as f:
        pickle.dump(result_dict, f)
    pd.to_pickle(
        {"config": config, "binned_df": binned_df},
        f"{fname.with_suffix('')}_binned_df.pkl",
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-f", "--file", type=str, help="File to process")

    main(parser.parse_args())
