"""
Surprisal extraction pipeline for scalar-implicature seminar project.
Items drawn from Breheny, Katsos & Williams (2006), Experiment 3,
Appendix A.3 -- the Implicit
Upper-Bound vs Implicit Lower-Bound conditions with trigger "some".
"""

import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Two models: GPT-2 small (124M params) and GPT-2 medium (355M params).
# Running both lets us check whether the surprisal pattern is specific to
# one model size or holds more generally across scale.
MODEL_NAMES = ["gpt2", "gpt2-medium"]

_tokenizers = {}
_models = {}
for name in MODEL_NAMES:
    _tokenizers[name] = GPT2TokenizerFast.from_pretrained(name)
    _models[name] = GPT2LMHeadModel.from_pretrained(name)
    _models[name].eval()


def word_surprisal(sentence: str, target_word: str, model_name: str = "gpt2") -> float:
    """Surprisal (bits) of the FIRST WHOLE-WORD occurrence of target_word,
    given context before it. Uses a word-boundary regex so short trigger
    words like "or" don't match substrings inside other words (e.g.
    "working", "for", "routine")."""
    tokenizer = _tokenizers[model_name]
    model = _models[model_name]
    pattern = r"\b" + re.escape(target_word) + r"\b"
    m = re.search(pattern, sentence, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"'{target_word}' not found as a whole word in: {sentence}")
    idx = m.start()

    prefix = sentence[:idx]
    full_ids = tokenizer.encode(sentence)
    prefix_ids = tokenizer.encode(prefix)
    n_prefix = len(prefix_ids)

    target_word_ids = tokenizer.encode(" " + target_word.strip())
    n_target_tokens = len(target_word_ids)
    target_ids = full_ids[n_prefix:n_prefix + n_target_tokens]

    input_ids = torch.tensor([full_ids])
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    total_bits = 0.0
    for i, tok_id in enumerate(target_ids):
        pos = n_prefix + i - 1
        if pos < 0:
            continue
        logp = log_probs[pos, tok_id].item()
        total_bits += -logp / torch.log(torch.tensor(2.0)).item()
    return total_bits


# ---------------------------------------------------------------------
# Items: Breheny, Katsos & Williams (2006), Experiment 3, Appendix A.3
# English translations of the original Greek stimuli, as published.
# Condition: implicit upper-bound (implicature warranted) vs.
# implicit lower-bound (implicature not warranted).
# Critical word: "some" (in "some of the Fs").
# ---------------------------------------------------------------------
SENTENCE_ITEMS = [
    (1, "upper_bound",
     "Mary asked John whether he intended to host all his relatives in his "
     "tiny apartment. John replied that he intended to host some of his "
     "relatives. The rest would stay in a nearby hotel.", "some"),
    (1, "lower_bound",
     "Mary was surprised to see John cleaning his apartment and she asked "
     "the reason why. John replied that he intended to host some of his "
     "relatives. The rest would stay in a nearby hotel.", "some"),

    (2, "upper_bound",
     "Mary wanted to create a very special atmosphere and wondered whether "
     "to light all her special candles in the dining hall or not. "
     "Eventually she took her decision: she lit some of her candles. "
     "The others were left unlit to create a nice atmosphere.", "some"),
    (2, "lower_bound",
     "Mary pays attention to the most minute detail in order to create a "
     "romantic atmosphere in her dinner parties. Last night she had a "
     "dinner party and she lit some of her candles. The others were left "
     "unlit to create a nice atmosphere.", "some"),

    (3, "upper_bound",
     "Yesterday, George asked Mary whether John had gone to the cinema "
     "with all his friends. Mary said that he went to the cinema with "
     "some of his friends. The others went for bowling and "
     "roller-skating.", "some"),
    (3, "lower_bound",
     "John's mother is anxious about his whereabouts. She asked Mary if "
     "she knew what John did on Friday night. Mary said that he went to "
     "the cinema with some of his friends. The others went for bowling "
     "and roller-skating.", "some"),

    (4, "upper_bound",
     "The warden of the zoo asked whether all the lions had been fed. "
     "The worker replied that he had fed some of the lions. The rest "
     "were being examined by the vets and would be fed later.", "some"),
    (4, "lower_bound",
     "The warden of the zoo asked the worker what he had been doing the "
     "whole morning. He replied that he had fed some of the lions. The "
     "rest were being examined by the vets and would be fed later.",
     "some"),

    # Appendix A.1 (Experiment 1) -- same paper, different trigger word "or"
    (5, "upper_bound",
     "John was taking a university course and working at the same time. "
     "For the exams he had to study from short and comprehensive sources. "
     "Depending on the course, he decided to read the class notes or the "
     "summary.", "or"),
    (5, "lower_bound",
     "John heard that the textbook for Geophysics was very advanced. "
     "Nobody understood it properly. He heard that if he wanted to pass "
     "the course he should read the class notes or the summary.", "or"),

    (6, "upper_bound",
     "The day's offer usually is: you can have a full menu for one person "
     "and the second person can have the plat de jour for free. Today, "
     "customers could have for free meat or fish.", "or"),
    (6, "lower_bound",
     "The dietician that visited the school explained to children how "
     "useful for our body protein can be. He also told them that we can "
     "find protein in meat or fish.", "or"),

    (7, "upper_bound",
     "Mary was saying that John is so careless that he constantly loses "
     "things. He has lost money, keys, credit cards. Once along with his "
     "bag, he even managed to lose his ID or his passport.", "or"),
    (7, "lower_bound",
     "The police stopped John for a routine control but John had only his "
     "driving license on him. The policemen stressed to John that valid "
     "identification documents are his ID or his passport.", "or"),

    (8, "upper_bound",
     "While Mary and John were out shopping, it started raining. John "
     "would get wet. Even though she did not have a lot of money, she "
     "offered to buy him an umbrella or a coat.", "or"),
    (8, "lower_bound",
     "It was highly probable that it would rain. Mary advised John to "
     "dress accordingly. To avoid getting wet, she suggested to him to "
     "take with him an umbrella or a coat.", "or"),
]


def run_all(items, model_name="gpt2"):
    rows = []
    for item_id, condition, sentence, crit_word in items:
        bits = word_surprisal(sentence, crit_word, model_name=model_name)
        rows.append({
            "item_id": item_id,
            "condition": condition,
            "sentence": sentence,
            "critical_word": crit_word,
            "model": model_name,
            "surprisal_bits": bits,
        })
    return pd.DataFrame(rows)


# Human benchmarks: Breheny et al. (2006), condition-level mean trigger
# reading times (ms). Two separate benchmarks since items 1-4 (trigger
# "some") are from Experiment 3, Table 5, and items 5-8 (trigger "or")
# are from Experiment 1. Report them separately, not pooled -- they come
# from different experiments/participant groups.
HUMAN_BENCHMARK_MS = {
    "some_trigger": {"upper_bound": 1027, "lower_bound": 927},   # Exp 3, Table 5
    "or_trigger":   {"upper_bound": 1291, "lower_bound": 1204},  # Exp 1, main text
}


if __name__ == "__main__":
    from scipy.stats import ttest_rel, wilcoxon

    all_dfs = []
    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}\nMODEL: {model_name}\n{'='*60}")
        df = run_all(SENTENCE_ITEMS, model_name=model_name)
        all_dfs.append(df)
        print(df[["item_id", "condition", "surprisal_bits"]])

        wide = df.pivot(index="item_id", columns="condition", values="surprisal_bits")
        print("\nPer-item surprisal (bits):")
        print(wide)

        t, p = ttest_rel(wide["upper_bound"], wide["lower_bound"])
        print(f"\nPaired t-test (all items): t={t:.3f}, p={p:.3f}")
        try:
            w, p_w = wilcoxon(wide["upper_bound"], wide["lower_bound"])
            print(f"Wilcoxon signed-rank: W={w:.3f}, p={p_w:.3f}")
        except ValueError as e:
            print(f"Wilcoxon not computable with this sample: {e}")

        print(f"\nMean surprisal (all items): upper={wide['upper_bound'].mean():.3f} bits, "
              f"lower={wide['lower_bound'].mean():.3f} bits")

        for trig, ids in [("some_trigger", [1, 2, 3, 4]), ("or_trigger", [5, 6, 7, 8])]:
            sub = wide.loc[wide.index.isin(ids)]
            print(f"\n[{trig}] mean surprisal: upper={sub['upper_bound'].mean():.3f} bits, "
                  f"lower={sub['lower_bound'].mean():.3f} bits")
            bench = HUMAN_BENCHMARK_MS[trig]
            print(f"[{trig}] human RT (Breheny et al. 2006): "
                  f"upper={bench['upper_bound']}ms, lower={bench['lower_bound']}ms")

    # Save combined results across both models
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("surprisal_results_both_models.csv", index=False)
    print("\nSaved combined results to surprisal_results_both_models.csv")

    # -------------------------------------------------------------
    # Plot: mean surprisal by trigger type and condition, one panel
    # per model. Suitable for dropping into the summary/poster.
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(10, 4.5), sharey=True)
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    trigger_groups = [("some", [1, 2, 3, 4]), ("or", [5, 6, 7, 8])]
    bar_width = 0.35
    x = np.arange(len(trigger_groups))

    for ax, model_name in zip(axes, MODEL_NAMES):
        df_m = combined[combined["model"] == model_name]
        wide_m = df_m.pivot(index="item_id", columns="condition",
                             values="surprisal_bits")

        upper_means, lower_means = [], []
        for _, ids in trigger_groups:
            sub = wide_m.loc[wide_m.index.isin(ids)]
            upper_means.append(sub["upper_bound"].mean())
            lower_means.append(sub["lower_bound"].mean())

        ax.bar(x - bar_width / 2, upper_means, bar_width, label="Upper-bound")
        ax.bar(x + bar_width / 2, lower_means, bar_width, label="Lower-bound")
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in trigger_groups])
        ax.set_title(model_name)
        ax.set_xlabel("Trigger word")

    axes[0].set_ylabel("Mean surprisal (bits)")
    axes[0].legend()
    fig.suptitle("GPT-2 Surprisal by Trigger Type, Condition, and Model Size")
    fig.tight_layout()
    fig.savefig("surprisal_plot.png", dpi=200, bbox_inches="tight")
    print("\nSaved plot to surprisal_plot.png")
    plt.show()
