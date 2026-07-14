"""WEAT/SEAT bias metrics implementation.

Based on:
- Caliskan et al. (2017) - WEAT
- May et al. (2019) - SEAT
- España-Bonet & Barrón-Cedeño (2022) - CA-WEAT
"""

import numpy as np
from scipy import stats
from typing import Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def mean_cosine_similarity(
    target: np.ndarray,
    attribute_set: np.ndarray,
) -> float:
    """Compute mean cosine similarity between a target and attribute set."""
    similarities = np.dot(attribute_set, target) / (
        np.linalg.norm(attribute_set, axis=1) * np.linalg.norm(target)
    )
    return float(np.mean(similarities))


def weat_statistic(
    target_x: np.ndarray,
    target_y: np.ndarray,
    attribute_a: np.ndarray,
    attribute_b: np.ndarray,
) -> float:
    """Compute WEAT test statistic.

    s(x, A, B) = mean(cos(x, a)) - mean(cos(x, b))
    Weat = sum(s(x, A, B) for x in X) - sum(s(y, A, B) for y in Y)
    """
    scores_x = []
    for x in target_x:
        sim_a = mean_cosine_similarity(x, attribute_a)
        sim_b = mean_cosine_similarity(x, attribute_b)
        scores_x.append(sim_a - sim_b)

    scores_y = []
    for y in target_y:
        sim_a = mean_cosine_similarity(y, attribute_a)
        sim_b = mean_cosine_similarity(y, attribute_b)
        scores_y.append(sim_a - sim_b)

    return sum(scores_x) - sum(scores_y)


def weat_effect_size(
    target_x: np.ndarray,
    target_y: np.ndarray,
    attribute_a: np.ndarray,
    attribute_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size for WEAT.

    d = (mean(s(x, A, B)) - mean(s(y, A, B))) / std(all s)
    """
    scores_x = []
    for x in target_x:
        sim_a = mean_cosine_similarity(x, attribute_a)
        sim_b = mean_cosine_similarity(x, attribute_b)
        scores_x.append(sim_a - sim_b)

    scores_y = []
    for y in target_y:
        sim_a = mean_cosine_similarity(y, attribute_a)
        sim_b = mean_cosine_similarity(y, attribute_b)
        scores_y.append(sim_a - sim_b)

    all_scores = scores_x + scores_y
    pooled_std = np.std(all_scores, ddof=1)

    if pooled_std == 0:
        return 0.0

    return float((np.mean(scores_x) - np.mean(scores_y)) / pooled_std)


def weat_effect_size_ci(
    target_x: np.ndarray,
    target_y: np.ndarray,
    attribute_a: np.ndarray,
    attribute_b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Cohen's d with bootstrap confidence interval.

    Returns:
        (effect_size, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    d_obs = weat_effect_size(target_x, target_y, attribute_a, attribute_b)
    all_scores_x = []
    for x in target_x:
        sim_a = mean_cosine_similarity(x, attribute_a)
        sim_b = mean_cosine_similarity(x, attribute_b)
        all_scores_x.append(sim_a - sim_b)

    all_scores_y = []
    for y in target_y:
        sim_a = mean_cosine_similarity(y, attribute_a)
        sim_b = mean_cosine_similarity(y, attribute_b)
        all_scores_y.append(sim_a - sim_b)

    scores_x = np.array(all_scores_x)
    scores_y = np.array(all_scores_y)
    n_x = len(scores_x)
    n_y = len(scores_y)

    boot_ds = []
    for _ in range(n_bootstrap):
        bx = scores_x[rng.integers(0, n_x, size=n_x)]
        by = scores_y[rng.integers(0, n_y, size=n_y)]
        pooled = np.concatenate([bx, by])
        p_std = np.std(pooled, ddof=1)
        if p_std > 0:
            boot_ds.append(float((np.mean(bx) - np.mean(by)) / p_std))
        else:
            boot_ds.append(0.0)

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(boot_ds, alpha * 100))
    ci_upper = float(np.percentile(boot_ds, (1 - alpha) * 100))
    return d_obs, ci_lower, ci_upper


def weat_p_value(
    target_x: np.ndarray,
    target_y: np.ndarray,
    attribute_a: np.ndarray,
    attribute_b: np.ndarray,
    n_permutations: int = 10000,
) -> float:
    """Compute p-value via permutation test.

    H0: no difference in association between target groups and attributes.
    """
    observed = weat_statistic(target_x, target_y, attribute_a, attribute_b)
    all_targets = np.vstack([target_x, target_y])
    n_x = len(target_x)
    count = 0

    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        perm = rng.permutation(all_targets)
        perm_x = perm[:n_x]
        perm_y = perm[n_x:]
        perm_stat = weat_statistic(perm_x, perm_y, attribute_a, attribute_b)
        if perm_stat >= observed:
            count += 1

    return count / n_permutations


def run_weat_test(
    embeddings_dict: dict[str, np.ndarray],
    target1_words: list[str],
    target2_words: list[str],
    attribute1_words: list[str],
    attribute2_words: list[str],
    n_permutations: int = 10000,
) -> dict:
    """Run a complete WEAT test.

    Args:
        embeddings_dict: mapping of word -> embedding vector
        target1_words: words in target group 1
        target2_words: words in target group 2
        attribute1_words: words in attribute group 1
        attribute2_words: words in attribute group 2
        n_permutations: number of permutations for p-value

    Returns:
        dict with effect_size, p_value, statistic, and details
    """
    def get_embeddings(words: list[str]) -> tuple[np.ndarray, list[str]]:
        found = []
        vectors = []
        for w in words:
            if w in embeddings_dict:
                found.append(w)
                vectors.append(embeddings_dict[w])
            else:
                print(f"  Warning: '{w}' not found in embeddings")
        return np.array(vectors), found

    target_x, found_x = get_embeddings(target1_words)
    target_y, found_y = get_embeddings(target2_words)
    attr_a, found_a = get_embeddings(attribute1_words)
    attr_b, found_b = get_embeddings(attribute2_words)

    if len(target_x) < 2 or len(target_y) < 2 or len(attr_a) < 2 or len(attr_b) < 2:
        return {
            "effect_size": None,
            "p_value": None,
            "statistic": None,
            "error": "Insufficient words found in embeddings",
            "found": {
                "target1": found_x,
                "target2": found_y,
                "attribute1": found_a,
                "attribute2": found_b,
            },
        }

    effect_size = weat_effect_size(target_x, target_y, attr_a, attr_b)
    d_obs, ci_lower, ci_upper = weat_effect_size_ci(target_x, target_y, attr_a, attr_b)
    p_value = weat_p_value(target_x, target_y, attr_a, attr_b, n_permutations)
    statistic = weat_statistic(target_x, target_y, attr_a, attr_b)

    return {
        "effect_size": effect_size,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "statistic": statistic,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_target1": len(found_x),
        "n_target2": len(found_y),
        "n_attribute1": len(found_a),
        "n_attribute2": len(found_b),
        "found": {
            "target1": found_x,
            "target2": found_y,
            "attribute1": found_a,
            "attribute2": found_b,
        },
    }


def run_seat_test(
    target1_embeddings: np.ndarray,
    target2_embeddings: np.ndarray,
    attribute1_embeddings: np.ndarray,
    attribute2_embeddings: np.ndarray,
    n_permutations: int = 10000,
) -> dict:
    """Run a SEAT test using pre-computed sentence embeddings.

    Args:
        target1_embeddings: Embeddings for target group 1 sentences.
        target2_embeddings: Embeddings for target group 2 sentences.
        attribute1_embeddings: Embeddings for attribute group 1.
        attribute2_embeddings: Embeddings for attribute group 2.
        n_permutations: Number of permutations for p-value.

    Returns:
        dict with effect_size, p_value, statistic, and details
    """
    if len(target1_embeddings) < 2 or len(target2_embeddings) < 2:
        return {
            "effect_size": None,
            "p_value": None,
            "statistic": None,
            "error": "Insufficient embeddings for SEAT test",
        }

    effect_size = weat_effect_size(
        target1_embeddings, target2_embeddings,
        attribute1_embeddings, attribute2_embeddings,
    )
    d_obs, ci_lower, ci_upper = weat_effect_size_ci(
        target1_embeddings, target2_embeddings,
        attribute1_embeddings, attribute2_embeddings,
    )
    p_value = weat_p_value(
        target1_embeddings, target2_embeddings,
        attribute1_embeddings, attribute2_embeddings,
        n_permutations,
    )
    statistic = weat_statistic(
        target1_embeddings, target2_embeddings,
        attribute1_embeddings, attribute2_embeddings,
    )

    return {
        "effect_size": effect_size,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "statistic": statistic,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_target1": len(target1_embeddings),
        "n_target2": len(target2_embeddings),
        "n_attribute1": len(attribute1_embeddings),
        "n_attribute2": len(attribute2_embeddings),
    }
