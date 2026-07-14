"""Analysis and visualization of bias experiment results."""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import MODELS


RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_results(filepath: str | Path) -> dict:
    """Load experiment results from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_bonferroni(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm-Bonferroni correction within each (model, language, group) group.

    Correction families per language:
    - weat_equivalent: the tests common across languages (n varies per language)
    - weat_ibge: 8 word-level IBGE racial category tests (PT-BR only)
    - seat_ibge: 8 sentence-level SEAT IBGE tests (PT-BR only)
    - regional: regionalism test (PT-BR only, n=1, no correction needed)

    For ES/FR/DE: n=7 (gender, 4 race, 2 sentiment)
    For EN: n=5 (gender, race adjectives, SEAT race professions, flowers/insects, instruments/weapons)
    For PT: n=5 (gender, race adjectives, SEAT race professions, flowers/insects, instruments/weapons) + IBGE + regionalism
    """
    df = df.copy()
    df["p_bonferroni"] = np.nan

    df["_bonf_group"] = df["test"].apply(
        lambda t: "seat_ibge" if t.startswith("SEAT_ibge")
        else "weat_ibge" if t.startswith("IBGE")
        else "regional" if "region" in t
        else "weat_equivalent"
    )

    for (model, lang, grp), group in df.groupby(["model", "language", "_bonf_group"]):
        n_tests = len(group)
        if n_tests == 0:
            continue
        idx = group.index
        sorted_pvals = np.sort(df.loc[idx, "p_value"].values)
        adjusted = np.empty(n_tests)
        for i, p in enumerate(sorted_pvals):
            adjusted[i] = min(p * (n_tests - i), 1.0)
        for i in range(1, n_tests):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])
        sorted_order = np.argsort(df.loc[idx, "p_value"].values)
        rank_order = np.argsort(sorted_order)
        df.loc[idx, "p_bonferroni"] = adjusted[rank_order]

    df["significant_bonferroni_005"] = df["p_bonferroni"] < 0.05
    df = df.drop(columns=["_bonf_group"])
    return df


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert experiment results to a pandas DataFrame."""
    rows = []
    experiments = results.get("experiments", [results])
    for exp in experiments:
        model = exp.get("model", "unknown")
        lang = exp.get("language", "unknown")
        for test_key, test_result in exp.get("tests", {}).items():
            if test_result.get("effect_size") is not None:
                dimension, test_name = test_key.split("/", 1)
                rows.append({
                    "model": model,
                    "language": lang,
                    "dimension": dimension,
                    "test": test_name,
                    "label": test_result.get("label", test_name),
                    "effect_size": test_result["effect_size"],
                    "ci_95_lower": test_result.get("ci_95_lower"),
                    "ci_95_upper": test_result.get("ci_95_upper"),
                    "p_value": test_result["p_value"],
                    "statistic": test_result.get("statistic"),
                    "significant_005": test_result.get("significant_005", False),
                    "significant_001": test_result.get("significant_001", False),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = apply_bonferroni(df)
    return df


def plot_bias_heatmap(df: pd.DataFrame, output_path: str | Path = None):
    """Plot heatmap of effect sizes across models and tests."""
    if df.empty:
        print("No data to plot")
        return

    languages = sorted(df["language"].unique())
    n_langs = len(languages)
    fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 6))
    if n_langs == 1:
        axes = [axes]

    lang_names = {"pt": "PT-BR", "en": "English", "es": "Spanish", "fr": "French", "de": "German"}

    for ax, lang in zip(axes, languages):
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            ax.set_title(f"{lang_names.get(lang, lang)} - No data")
            continue

        pivot = lang_df.pivot_table(
            index="model",
            columns="label",
            values="effect_size",
            aggfunc="first",
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-2,
            vmax=2,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Cohen's d"},
        )
        ax.set_title(f"Bias Effect Sizes ({lang_names.get(lang, lang)})")
        ax.set_xlabel("")
        ax.set_ylabel("Model")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.suptitle("WEAT Bias Evaluation: Effect Sizes Across Models", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {output_path}")
    plt.close(fig)


def plot_effect_size_comparison(df: pd.DataFrame, output_path: str | Path = None):
    """Plot grouped bar chart of effect sizes by dimension."""
    if df.empty:
        print("No data to plot")
        return

    grouped = df.groupby(["model", "dimension", "language"])["effect_size"].mean().reset_index()
    languages = sorted(df["language"].unique())
    n_langs = len(languages)
    lang_names = {"pt": "PT-BR", "en": "English", "es": "Spanish", "fr": "French", "de": "German"}

    fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 6))
    if n_langs == 1:
        axes = [axes]

    for ax, lang in zip(axes, languages):
        lang_data = grouped[grouped["language"] == lang]
        if lang_data.empty:
            ax.set_title(f"{lang_names.get(lang, lang)} - No data")
            continue

        pivot = lang_data.pivot_table(
            index="model",
            columns="dimension",
            values="effect_size",
            aggfunc="first",
        )

        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"{lang_names.get(lang, lang)} - Mean Effect Size by Dimension")
        ax.set_ylabel("Mean Cohen's d")
        ax.set_xlabel("")
        ax.legend(title="Dimension", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    plt.suptitle("Bias Dimensions Comparison Across Models", fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {output_path}")
    plt.close(fig)


def plot_significance_matrix(df: pd.DataFrame, output_path: str | Path = None):
    """Plot significance matrix showing which tests are significant."""
    if df.empty:
        print("No data to plot")
        return

    languages = sorted(df["language"].unique())
    n_langs = len(languages)
    lang_names = {"pt": "PT-BR", "en": "English", "es": "Spanish", "fr": "French", "de": "German"}

    fig, axes = plt.subplots(1, n_langs, figsize=(5 * n_langs, 5))
    if n_langs == 1:
        axes = [axes]

    for ax, lang in zip(axes, languages):
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            ax.set_title(f"{lang_names.get(lang, lang)} - No data")
            continue

        pivot = lang_df.pivot_table(
            index="model",
            columns="label",
            values="significant_005",
            aggfunc="first",
        )
        pivot = pivot.astype(float)

        sns.heatmap(
            pivot,
            annot=True,
            fmt="",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Significant (p<0.05)"},
        )
        ax.set_title(f"{lang_names.get(lang, lang)} - Statistical Significance")
        ax.set_xlabel("")
        ax.set_ylabel("Model")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.suptitle("Significance Matrix: Tests with p < 0.05", fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Significance matrix saved to: {output_path}")
    plt.close(fig)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table of effect sizes with CIs and Bonferroni-corrected p-values."""
    if df.empty:
        return "% No data available"

    pivot = df.pivot_table(
        index="model",
        columns="label",
        values="effect_size",
        aggfunc="first",
    )

    ci_lower = df.pivot_table(
        index="model",
        columns="label",
        values="ci_95_lower",
        aggfunc="first",
    )

    ci_upper = df.pivot_table(
        index="model",
        columns="label",
        values="ci_95_upper",
        aggfunc="first",
    )

    sig_pivot = df.pivot_table(
        index="model",
        columns="label",
        values="p_bonferroni",
        aggfunc="first",
    )

    n_cols = len(pivot.columns)
    col_spec = "l" + "c" * n_cols

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{WEAT Effect Sizes (Cohen's $d$) with 95\\% Bootstrap CI and Bonferroni-Corrected $p$-values}")
    lines.append("\\label{tab:effect_sizes}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Model"
    for col in pivot.columns:
        short = col.replace(": ", "\\newline ")
        header += f" & {short}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    for model in pivot.index:
        row = model.replace("-", "\\_")
        for col in pivot.columns:
            val = pivot.loc[model, col] if model in pivot.index and col in pivot.columns else None
            lo = ci_lower.loc[model, col] if model in ci_lower.index and col in ci_lower.columns else None
            hi = ci_upper.loc[model, col] if model in ci_upper.index and col in ci_upper.columns else None
            pval = sig_pivot.loc[model, col] if model in sig_pivot.index and col in sig_pivot.columns else None

            if pd.isna(val):
                row += " & --"
            else:
                cell = f"{val:.3f}"
                if not (pd.isna(lo) or pd.isna(hi)):
                    cell += f" [{lo:.2f}, {hi:.2f}]"
                if pval is not None and pval < 0.001:
                    cell = f"\\textbf{{{cell}}}$^{{***}}$"
                elif pval is not None and pval < 0.01:
                    cell = f"\\textbf{{{cell}}}$^{{**}}$"
                elif pval is not None and pval < 0.05:
                    cell = f"\\textbf{{{cell}}}$^{{*}}$"
                row += f" & {cell}"
        lines.append(row + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{2mm}")
    lines.append("\\footnotesize{$^* p_{\\mathrm{bonf}} < 0.05$, $^{**} p_{\\mathrm{bonf}} < 0.01$, $^{***} p_{\\mathrm{bonf}} < 0.001$. Brackets show 95\\% bootstrap CI.}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def print_summary(df: pd.DataFrame):
    """Print a human-readable summary of results."""
    if df.empty:
        print("No results to summarize.")
        return

    print("\n" + "=" * 90)
    print("BIAS EVALUATION SUMMARY (with Bonferroni correction)")
    print("=" * 90)

    lang_names = {"pt": "PT-BR", "en": "EN", "es": "ES", "fr": "FR", "de": "DE"}
    for (model, lang), group in df.groupby(["model", "language"]):
        lang_label = lang_names.get(lang, lang.upper())
        n_tests = len(group)
        print(f"\n{model} ({lang_label}) — {n_tests} tests:")
        for _, row in group.iterrows():
            sig_b = "***" if row.get("significant_bonferroni_005", False) else ""
            ci_str = ""
            if not pd.isna(row.get("ci_95_lower")) and not pd.isna(row.get("ci_95_upper")):
                ci_str = f"  95% CI [{row['ci_95_lower']:.3f}, {row['ci_95_upper']:.3f}]"
            print(f"  {row['label']:40s} d={row['effect_size']:+.4f}  p={row['p_value']:.4f}  p_bonf={row.get('p_bonferroni', row['p_value']):.4f}{sig_b}{ci_str}")


def compute_model_bias_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean absolute effect size per model across all tests."""
    summary = df.groupby("model").agg(
        mean_abs_effect=("effect_size", lambda x: np.mean(np.abs(x))),
        std_effect=("effect_size", "std"),
        n_tests=("effect_size", "count"),
        n_significant_bonf=("significant_bonferroni_005", "sum"),
    ).reset_index()

    model_info = []
    for model in summary["model"]:
        info = MODELS.get(model, {})
        model_info.append({
            "model": model,
            "parameters": info.get("parameters"),
            "dimensions": info.get("dimensions"),
            "type": info.get("type", "unknown"),
        })
    model_df = pd.DataFrame(model_info)

    return summary.merge(model_df, on="model")


def plot_size_vs_bias(df: pd.DataFrame, output_path: str | Path = None):
    """Plot scatter: model size (parameters or dimensions) vs mean absolute bias."""
    summary = compute_model_bias_summary(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"commercial": "#1f77b4", "open-source": "#2ca02c", "unknown": "#999999"}

    for ax, x_col, x_label in [
        (axes[0], "parameters", "Number of Parameters"),
        (axes[1], "dimensions", "Embedding Dimension"),
    ]:
        plot_df = summary.dropna(subset=[x_col])
        if plot_df.empty:
            ax.set_title(f"No data for {x_label}")
            continue

        for model_type in ["commercial", "open-source", "type"]:
            subset = plot_df[plot_df["type"] == model_type] if model_type != "type" else plot_df
            if subset.empty:
                continue
            color = colors.get(model_type, "#999999")
            marker = "o" if model_type == "commercial" else "s"
            label = model_type.capitalize() if model_type != "type" else "All"
            ax.scatter(subset[x_col], subset["mean_abs_effect"], c=color, marker=marker,
                       s=100, label=label, edgecolors="black", linewidth=0.5)

        for _, row in plot_df.iterrows():
            ax.annotate(row["model"], (row[x_col], row["mean_abs_effect"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        valid = plot_df.dropna(subset=[x_col, "mean_abs_effect"])
        if len(valid) >= 3:
            corr_spearman, p_spearman = stats.spearmanr(valid[x_col], valid["mean_abs_effect"])
            corr_pearson, p_pearson = stats.pearsonr(valid[x_col], valid["mean_abs_effect"])

            x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
            slope, intercept = np.polyfit(valid[x_col], valid["mean_abs_effect"], 1)
            ax.plot(x_range, slope * x_range + intercept, "--", color="gray", alpha=0.5)

            stats_text = f"Spearman r={corr_spearman:.3f} (p={p_spearman:.3f})\nPearson r={corr_pearson:.3f} (p={p_pearson:.3f})"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean |Cohen's d|")
        ax.set_title(f"Bias vs {x_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Model Size and Bias: Does Bigger Mean Fairer?", fontsize=14)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Size vs bias plot saved to: {output_path}")
    plt.close(fig)


def compute_size_bias_correlation(df: pd.DataFrame) -> dict:
    """Compute correlation statistics between model size and bias."""
    summary = compute_model_bias_summary(df)
    results = {}

    for x_col in ["parameters", "dimensions"]:
        valid = summary.dropna(subset=[x_col])
        if len(valid) >= 3:
            corr_spearman, p_spearman = stats.spearmanr(valid[x_col], valid["mean_abs_effect"])
            corr_pearson, p_pearson = stats.pearsonr(valid[x_col], valid["mean_abs_effect"])
            results[x_col] = {
                "spearman_r": corr_spearman,
                "spearman_p": p_spearman,
                "pearson_r": corr_pearson,
                "pearson_p": p_pearson,
                "n_models": len(valid),
            }
    return results


def main():
    """Generate all analysis artifacts."""
    # Load results
    results = load_results(RESULTS_DIR / "summary.json")
    df = results_to_dataframe(results)

    print(f"Loaded {len(df)} test results")

    # Print summary
    print_summary(df)

    # Generate plots
    plot_bias_heatmap(df, FIGURES_DIR / "heatmap.png")
    plot_effect_size_comparison(df, FIGURES_DIR / "effect_sizes.png")
    plot_significance_matrix(df, FIGURES_DIR / "significance.png")
    plot_size_vs_bias(df, FIGURES_DIR / "size_vs_bias.png")

    # Compute size-bias correlation
    corr_results = compute_size_bias_correlation(df)
    print("\nSize-Bias Correlation:")
    for col, stats_dict in corr_results.items():
        print(f"  {col}: Spearman r={stats_dict['spearman_r']:.3f} (p={stats_dict['spearman_p']:.3f})")

    # Generate LaTeX table
    latex = generate_latex_table(df)
    latex_path = RESULTS_DIR / "effect_sizes_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")

    # Save CSV for reference
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
