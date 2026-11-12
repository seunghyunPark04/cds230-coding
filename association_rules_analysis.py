#!/usr/bin/env python3
"""Explore Thanksgiving dish data and mine association rules.

Usage:
    python -m pip install -r requirements.txt
    python association_rules_analysis.py thanksgiving_dishes_onehot.csv

The CSV is expected to contain one row per respondent, a RespondentID column,
then one 0/1 column per dish. The respondent ID is never treated as a dish.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a one-hot Thanksgiving dataset and mine association rules."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="thanksgiving_dishes_onehot.csv",
        help="Input CSV path (default: thanksgiving_dishes_onehot.csv)",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Minimum share of respondents containing an itemset (default: 0.05)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Minimum rule confidence (default: 0.50)",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=1.0,
        help="Keep rules with lift at least this value (default: 1.0)",
    )
    return parser.parse_args()


def validate_thresholds(args: argparse.Namespace) -> None:
    if not 0 < args.min_support <= 1:
        raise ValueError("--min-support must be greater than 0 and at most 1.")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1.")
    if args.min_lift < 0:
        raise ValueError("--min-lift cannot be negative.")


def inspect_and_prepare(csv_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Print dataset diagnostics and return validated 0/1 dish columns."""
    if not csv_file.is_file():
        raise FileNotFoundError(
            f"Could not find {csv_file}. Put the CSV beside this script, "
            "or pass its path as the first argument."
        )

    # Step 1: load and inspect the raw data.
    data = pd.read_csv(csv_file)
    data.columns = data.columns.astype(str).str.strip()
    print("\n=== Dataset inspection ===")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print("Column names:")
    print(data.columns.tolist())
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values by column:")
    missing = data.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicated}")

    id_column = "RespondentID"
    if id_column not in data.columns:
        raise ValueError(f"Expected an ID column named {id_column!r}.")
    dish_columns = [column for column in data.columns if column != id_column]
    if not dish_columns:
        raise ValueError("No dish columns found after excluding RespondentID.")

    # Step 2: normalize item values and check that they really are one-hot indicators.
    original_items = data[dish_columns]
    numeric_items = original_items.apply(
        lambda column: pd.to_numeric(column.astype("string").str.strip(), errors="coerce")
    )
    invalid_text = original_items.notna() & numeric_items.isna()
    if invalid_text.any().any():
        bad_columns = invalid_text.any()[invalid_text.any()].index.tolist()
        raise ValueError(f"Non-numeric values found in dish columns: {bad_columns}")

    invalid_values = ~numeric_items.isna() & ~numeric_items.isin([0, 1])
    if invalid_values.any().any():
        bad_columns = invalid_values.any()[invalid_values.any()].index.tolist()
        raise ValueError(f"Dish columns must contain only 0, 1, or missing values: {bad_columns}")

    # Missing item responses mean unknown, not 'no'; exclude those rows rather than
    # silently turning missing values into zeros. Report how many are excluded.
    rows_before = len(numeric_items)
    complete_items = numeric_items.dropna(axis=0, how="any").astype(bool)
    dropped = rows_before - len(complete_items)
    print(f"\nRows excluded for missing dish values: {dropped}")
    if complete_items.empty:
        raise ValueError("No complete rows remain after removing missing dish values.")

    print(f"Rows used for association analysis: {len(complete_items)}")
    print(f"Dish columns used: {len(dish_columns)} (RespondentID excluded)")
    print("\nDish columns (boolean 0/1):")
    print(", ".join(dish_columns))
    return complete_items, dish_columns


def save_and_print_results(
    transactions: pd.DataFrame,
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save results, print useful summaries, and create two plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    itemsets_to_save = frequent_itemsets.copy()
    if not itemsets_to_save.empty:
        itemsets_to_save["itemsets"] = itemsets_to_save["itemsets"].apply(
            lambda items: ", ".join(sorted(items))
        )
    itemsets_to_save.to_csv(output_dir / "frequent_itemsets.csv", index=False)

    rules_to_save = rules.copy()
    if not rules_to_save.empty:
        for column in ("antecedents", "consequents"):
            rules_to_save[column] = rules_to_save[column].apply(
                lambda items: ", ".join(sorted(items))
            )
    rules_to_save.to_csv(output_dir / "association_rules.csv", index=False)

    print("\n=== Association analysis ===")
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")
    print(f"Rules meeting the chosen thresholds: {len(rules)}")
    if rules.empty:
        print(
            "No rules met the thresholds. Try lowering --min-support or "
            "--min-confidence, or check that the CSV contains enough responses."
        )
    else:
        display_columns = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
        ]
        print("\nTop 10 rules by lift:")
        print(
            rules.sort_values(["lift", "confidence"], ascending=False)
            .loc[:, display_columns]
            .head(10)
            .to_string(index=False, formatters={
                "support": "{:.3f}".format,
                "confidence": "{:.3f}".format,
                "lift": "{:.3f}".format,
            })
        )
        best = rules.sort_values("lift", ascending=False).iloc[0]
        left = ", ".join(sorted(best["antecedents"]))
        right = ", ".join(sorted(best["consequents"]))
        print(
            f"\nExample strongest-lift rule: If a respondent selected [{left}], "
            f"the rule associates it with [{right}]. Its confidence is "
            f"{best['confidence']:.1%}, and its lift is {best['lift']:.2f}."
        )
        print(
            f"Rules with lift > 1: {(rules['lift'] > 1).sum()} of {len(rules)}."
        )

    # Plot 1: most commonly selected dishes.
    item_counts = transactions.sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, 6))
    item_counts.sort_values().plot.barh(ax=ax, color="darkorange")
    ax.set_title("15 Most Common Thanksgiving Dishes")
    ax.set_xlabel("Number of respondents selecting dish")
    ax.set_ylabel("Dish")
    fig.tight_layout()
    fig.savefig(output_dir / "top_dishes.png", dpi=150)
    plt.close(fig)

    # Plot 2: support-confidence map of rules; marker size indicates lift.
    fig, ax = plt.subplots(figsize=(9, 6))
    if not rules.empty:
        marker_sizes = 30 + 50 * rules["lift"].clip(lower=0, upper=5)
        scatter = ax.scatter(
            rules["support"],
            rules["confidence"],
            s=marker_sizes,
            c=rules["lift"],
            cmap="viridis",
            alpha=0.75,
            edgecolors="black",
            linewidths=0.3,
        )
        fig.colorbar(scatter, ax=ax, label="Lift")
    else:
        ax.text(
            0.5,
            0.5,
            "No rules met the selected thresholds",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("Association Rules: Support vs. Confidence")
    ax.set_xlabel("Support (share of all respondents)")
    ax.set_ylabel("Confidence (share of antecedent respondents)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rules_support_confidence.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved results and plots in: {output_dir.resolve()}")
    print("  frequent_itemsets.csv")
    print("  association_rules.csv")
    print("  top_dishes.png")
    print("  rules_support_confidence.png")


def main() -> None:
    args = parse_args()
    validate_thresholds(args)
    csv_file = Path(args.csv_file)
    transactions, _ = inspect_and_prepare(csv_file)

    # Step 3: find frequent combinations with FP-Growth.
    # Each row is a respondent/basket and each True cell means that dish was selected.
    print("\nFitting FP-Growth frequent-itemset model...")
    frequent_itemsets = fpgrowth(
        transactions,
        min_support=args.min_support,
        use_colnames=True,
    )

    # Step 4: turn frequent itemsets into rules and retain positively associated rules.
    if not frequent_itemsets.empty and frequent_itemsets["itemsets"].map(len).gt(1).any():
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=args.min_confidence,
        )
        rules = rules.loc[rules["lift"] >= args.min_lift].copy()
    else:
        rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )

    print(
        f"Thresholds: support >= {args.min_support:.3f}, "
        f"confidence >= {args.min_confidence:.3f}, lift >= {args.min_lift:.3f}"
    )
    save_and_print_results(
        transactions,
        frequent_itemsets,
        rules,
        output_dir=Path("association_results"),
    )


if __name__ == "__main__":
    main()
