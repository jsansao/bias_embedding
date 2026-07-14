"""Main experiment runner: extract embeddings and measure bias.

Usage:
    python run_experiments.py --language pt --model openai-small
    python run_experiments.py --language en --model gemini-001
    python run_experiments.py --language all --model all
"""

import argparse
import json
import itertools
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from embeddings import EmbeddingExtractor, MODELS, load_weat_lists
from weat import run_weat_test, run_seat_test

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent
WEAT_DIR = PROJECT_ROOT / "data" / "weat_lists"
RESULTS_DIR = PROJECT_ROOT / "results"


ALL_LANGUAGES = ["pt", "en", "es", "fr", "de"]

LANG_FILE_MAP = {
    "pt": "weat_ptbr.json",
    "en": "weat_en.json",
    "es": "weat_es.json",
    "fr": "weat_fr.json",
    "de": "weat_de.json",
}


def collect_words_for_language(weat_data: dict, language: str) -> list[str]:
    """Collect all unique words for a language across all test dimensions."""
    all_words = set()
    if language == "pt":
        dimensions = ["gender", "race_ethnicity", "race_ibge", "regionalism", "sentiment"]
    else:
        dimensions = ["gender", "race_ethnicity", "sentiment"]

    for dim in dimensions:
        dim_data = weat_data.get(dim, {})
        for test_name, test in dim_data.items():
            if not isinstance(test, dict):
                continue
            for key in ["target1", "target2", "attribute1", "attribute2"]:
                if key in test and isinstance(test[key], dict) and "words" in test[key]:
                    all_words.update(test[key]["words"])

    return sorted(all_words)


def collect_seat_sentences(weat_data: dict, language: str) -> dict[str, list[str]]:
    """Collect all SEAT sentences for a language.

    Returns:
        dict mapping test_key -> list of sentences to embed
    """
    sentences = {}
    if language != "pt":
        return sentences

    seat_tests = weat_data.get("race_ibge", {})
    for test_name, test in seat_tests.items():
        if not test_name.startswith("SEAT_"):
            continue
        if "target1_template" not in test:
            continue

        test_sentences = []
        template1 = test["target1_template"]
        template2 = test["target2_template"]
        targets = test["targets"]
        target_keys = list(targets.keys())

        for key in target_keys:
            for w in targets[key]:
                test_sentences.append(template1.format(word=w))
        for w in test["attribute1"]["words"]:
            test_sentences.append(w)
        for w in test["attribute2"]["words"]:
            test_sentences.append(w)

        sentences[f"race_ibge/{test_name}"] = test_sentences

    return sentences


def get_test_configs(language: str) -> list[tuple[str, str, str]]:
    """Get test configurations for a language (only full WEAT-format tests)."""
    if language == "pt":
        return [
            ("gender", "WEAT6_professions", "Gender: professions"),
            ("race_ethnicity", "WEAT_race_adjectives", "Race: adjectives"),
            ("race_ethnicity", "SEAT_race_professions", "Race: professions (SEAT)"),
            ("race_ibge", "IBGE_sentiment_amarela", "IBGE: branca/amarela (sent.)"),
            ("race_ibge", "IBGE_sentiment_parda", "IBGE: branca/parda (sent.)"),
            ("race_ibge", "IBGE_sentiment_preta", "IBGE: branca/preta (sent.)"),
            ("race_ibge", "IBGE_sentiment_indigena", "IBGE: branca/indígena (sent.)"),
            ("race_ibge", "IBGE_status_amarela", "IBGE: branca/amarela (status)"),
            ("race_ibge", "IBGE_status_parda", "IBGE: branca/parda (status)"),
            ("race_ibge", "IBGE_status_preta", "IBGE: branca/preta (status)"),
            ("race_ibge", "IBGE_status_indigena", "IBGE: branca/indígena (status)"),
            ("regionalism", "WEAT_region_adjectives", "Regionalism: adjectives"),
            ("sentiment", "WEAT1_flowers_insects", "Sentiment: flowers/insects"),
            ("sentiment", "WEAT2_instruments_weapons", "Sentiment: instruments/weapons"),
        ]
    elif language == "en":
        return [
            ("gender", "WEAT6_professions", "Gender: professions"),
            ("race_ethnicity", "WEAT_race_adjectives", "Race: adjectives"),
            ("race_ethnicity", "SEAT_race_professions", "Race: professions (SEAT)"),
            ("sentiment", "WEAT1_flowers_insects", "Sentiment: flowers/insects"),
            ("sentiment", "WEAT2_instruments_weapons", "Sentiment: instruments/weapons"),
        ]
    elif language == "de":
        return [
            ("gender", "WEAT6_professions", "Gender: professions"),
            ("race_ethnicity", "WEAT_race_turkish", "Race: Turkish"),
            ("race_ethnicity", "WEAT_race_arab", "Race: Arab"),
            ("race_ethnicity", "WEAT_race_african", "Race: African"),
            ("race_ethnicity", "WEAT_race_easteuropean", "Race: Eastern European"),
            ("sentiment", "WEAT1_flowers_insects", "Sentiment: flowers/insects"),
            ("sentiment", "WEAT2_instruments_weapons", "Sentiment: instruments/weapons"),
        ]
    elif language == "fr":
        return [
            ("gender", "WEAT6_professions", "Gender: professions"),
            ("race_ethnicity", "WEAT_race_maghrebi", "Race: Maghrebi"),
            ("race_ethnicity", "WEAT_race_subsaharan", "Race: Sub-Saharan"),
            ("race_ethnicity", "WEAT_race_muslim", "Race: Muslim"),
            ("race_ethnicity", "WEAT_race_asian", "Race: Asian"),
            ("sentiment", "WEAT1_flowers_insects", "Sentiment: flowers/insects"),
            ("sentiment", "WEAT2_instruments_weapons", "Sentiment: instruments/weapons"),
        ]
    elif language == "es":
        return [
            ("gender", "WEAT6_professions", "Gender: professions"),
            ("race_ethnicity", "WEAT_race_moroccan", "Race: Moroccan"),
            ("race_ethnicity", "WEAT_race_latino", "Race: Latin American"),
            ("race_ethnicity", "WEAT_race_roma", "Race: Roma"),
            ("race_ethnicity", "WEAT_race_subsaharan", "Race: Sub-Saharan"),
            ("sentiment", "WEAT1_flowers_insects", "Sentiment: flowers/insects"),
            ("sentiment", "WEAT2_instruments_weapons", "Sentiment: instruments/weapons"),
        ]


def run_bias_tests(
    model_key: str,
    language: str,
    embeddings_dict: dict,
    weat_data: dict,
    n_permutations: int = 5000,
) -> dict:
    """Run all WEAT bias tests for a given model and language."""
    results = {"model": model_key, "language": language, "tests": {}}
    test_configs = get_test_configs(language)

    for dimension, test_name, label in test_configs:
        test_data = weat_data[dimension][test_name]

        # Skip if missing required groups
        required = ["target1", "target2", "attribute1", "attribute2"]
        if not all(k in test_data for k in required):
            print(f"  {label}: SKIPPED (missing groups)")
            continue

        test_result = run_weat_test(
            embeddings_dict=embeddings_dict,
            target1_words=test_data["target1"]["words"],
            target2_words=test_data["target2"]["words"],
            attribute1_words=test_data["attribute1"]["words"],
            attribute2_words=test_data["attribute2"]["words"],
            n_permutations=n_permutations,
        )

        results["tests"][f"{dimension}/{test_name}"] = {
            "label": label,
            **test_result,
        }

        if test_result["effect_size"] is not None:
            sig = "***" if test_result["significant_001"] else ("*" if test_result["significant_005"] else "ns")
            print(f"  {label}: d={test_result['effect_size']:+.4f}, p={test_result['p_value']:.4f} ({sig})")
        else:
            print(f"  {label}: Error - {test_result.get('error', 'unknown')}")

    return results


def run_seat_bias_tests(
    model_key: str,
    language: str,
    seat_embeddings: dict[str, np.ndarray],
    weat_data: dict,
    n_permutations: int = 5000,
) -> dict:
    """Run SEAT bias tests for a given model and language.

    Compares branca (reference) against each other racial group.
    """
    results = {"model": model_key, "language": language, "tests": {}}
    seat_tests = weat_data.get("race_ibge", {})

    for test_name, test in seat_tests.items():
        if not test_name.startswith("SEAT_"):
            continue
        if "target1_template" not in test:
            continue

        test_key_base = f"race_ibge/{test_name}"
        if test_key_base not in seat_embeddings:
            print(f"  SEAT {test_name}: SKIPPED (no embeddings)")
            continue

        embs = seat_embeddings[test_key_base]
        targets = test["targets"]
        target_keys = list(targets.keys())
        if len(target_keys) < 2:
            continue

        ref_key = target_keys[0]
        ref_words = targets[ref_key]
        template = test["target1_template"]
        ref_sentences = [template.format(word=w) for w in ref_words]

        for other_key in target_keys[1:]:
            other_words = targets[other_key]
            other_sentences = [template.format(word=w) for w in other_words]

            ref_embs = np.array([embs[s] for s in ref_sentences if s in embs])
            other_embs = np.array([embs[s] for s in other_sentences if s in embs])
            attr1_embs = np.array([embs[w] for w in test["attribute1"]["words"] if w in embs])
            attr2_embs = np.array([embs[w] for w in test["attribute2"]["words"] if w in embs])

            test_key = f"{test_key_base}_{ref_key}_{other_key}"
            test_type = test_name.split("_", 2)[-1]
            label = f"SEAT IBGE: {ref_key}/{other_key} ({test_type})"

            test_result = run_seat_test(
                target1_embeddings=ref_embs,
                target2_embeddings=other_embs,
                attribute1_embeddings=attr1_embs,
                attribute2_embeddings=attr2_embs,
                n_permutations=n_permutations,
            )

            results["tests"][test_key] = {
                "label": label,
                **test_result,
            }

            if test_result["effect_size"] is not None:
                sig = "***" if test_result["significant_001"] else ("*" if test_result["significant_005"] else "ns")
                print(f"  {label}: d={test_result['effect_size']:+.4f}, p={test_result['p_value']:.4f} ({sig})")
            else:
                print(f"  {label}: Error - {test_result.get('error', 'unknown')}")

    return results


def save_results(results: dict, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def serialize(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=serialize)

    print(f"  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run WEAT bias experiments")
    parser.add_argument(
        "--language",
        choices=ALL_LANGUAGES + ["all"],
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    languages = ALL_LANGUAGES if args.language == "all" else [args.language]

    extractor = EmbeddingExtractor(max_workers=args.workers)

    # Load WEAT data
    weat_data = {}
    for lang in languages:
        weat_file = WEAT_DIR / LANG_FILE_MAP[lang]
        weat_data[lang] = load_weat_lists(str(weat_file))

    # Phase 1: Collect words — use composite key "model_lang"
    print("=" * 60)
    print("Phase 1: Collecting words for embedding extraction")
    print("=" * 60)

    tasks = {}
    for model_key, lang in itertools.product(models, languages):
        composite_key = f"{model_key}_{lang}"
        unique_words = collect_words_for_language(weat_data[lang], lang)
        tasks[composite_key] = unique_words
        print(f"  {composite_key}: {len(unique_words)} unique words")

    # Phase 1b: Collect SEAT sentences (PT-BR only)
    seat_tasks = {}
    seat_test_configs = {}
    if "pt" in languages:
        print("\n" + "=" * 60)
        print("Phase 1b: Collecting SEAT sentences")
        print("=" * 60)
        for model_key in models:
            composite_key = f"{model_key}_pt_seat"
            seat_sentences = collect_seat_sentences(weat_data["pt"], "pt")
            if seat_sentences:
                all_sentences = []
                for test_key, sents in seat_sentences.items():
                    all_sentences.extend(sents)
                seat_tasks[composite_key] = list(set(all_sentences))
                seat_test_configs[composite_key] = seat_sentences
                print(f"  {composite_key}: {len(seat_tasks[composite_key])} unique sentences")

    # Phase 2: Extract embeddings in parallel
    print("\n" + "=" * 60)
    print("Phase 2: Extracting embeddings (parallel)")
    print("=" * 60)

    start_time = time.time()
    all_embeddings = extractor.get_embeddings_parallel(tasks)
    elapsed = time.time() - start_time
    print(f"\nEmbedding extraction completed in {elapsed:.1f}s")

    # Phase 2b: Extract SEAT embeddings (PT-BR only)
    seat_all_embeddings = {}
    if seat_tasks:
        print("\n" + "=" * 60)
        print("Phase 2b: Extracting SEAT embeddings (parallel)")
        print("=" * 60)
        start_time = time.time()
        seat_all_embeddings = extractor.get_embeddings_parallel(seat_tasks)
        elapsed = time.time() - start_time
        print(f"\nSEAT embedding extraction completed in {elapsed:.1f}s")

    # Phase 3: Run WEAT tests
    print("\n" + "=" * 60)
    print("Phase 3: Running WEAT tests")
    print("=" * 60)

    all_results = []
    for model_key, lang in itertools.product(models, languages):
        composite_key = f"{model_key}_{lang}"
        print(f"\n--- {model_key} / {lang} ---")
        embeddings_dict = dict(zip(tasks[composite_key], all_embeddings[composite_key]))
        print(f"  Loaded {len(embeddings_dict)} embeddings")

        results = run_bias_tests(
            model_key, lang, embeddings_dict, weat_data[lang], args.permutations
        )
        all_results.append(results)
        output_file = RESULTS_DIR / f"{model_key}_{lang}.json"
        save_results(results, output_file)

    # Phase 4: Run SEAT tests (PT-BR only)
    if seat_all_embeddings:
        print("\n" + "=" * 60)
        print("Phase 4: Running SEAT tests")
        print("=" * 60)

        for model_key in models:
            composite_key = f"{model_key}_pt_seat"
            if composite_key not in seat_all_embeddings:
                continue

            print(f"\n--- {model_key} / PT-BR (SEAT) ---")
            seat_embs_array = seat_all_embeddings[composite_key]
            seat_sents = seat_tasks[composite_key]
            seat_embs_dict = dict(zip(seat_sents, seat_embs_array))

            seat_embeddings_by_test = {}
            for test_key, sents in seat_test_configs[composite_key].items():
                seat_embeddings_by_test[test_key] = seat_embs_dict

            seat_results = run_seat_bias_tests(
                model_key, "pt", seat_embeddings_by_test, weat_data["pt"], args.permutations
            )

            for existing_results in all_results:
                if existing_results["model"] == model_key and existing_results["language"] == "pt":
                    existing_results["tests"].update(seat_results["tests"])
                    output_file = RESULTS_DIR / f"{model_key}_pt.json"
                    save_results(existing_results, output_file)
                    break

    # Save combined summary
    summary_file = RESULTS_DIR / "summary.json"
    save_results({"experiments": all_results}, summary_file)
    print(f"\nAll results saved to: {summary_file}")


if __name__ == "__main__":
    main()
