"""Experiment: Perturbing 4 randomly selected variables with alpha in {-5.0, -2.5, -0.5, 0.5, 2.5, 5.0}.

This script:
1. Loads the 20 Newsgroups dataset and encodes sample texts using GTR-Base (d=768).
2. Randomly selects 4 variables (latent dimensions).
3. For each variable, applies perturbations with alpha in {-5.0, -2.5, -0.5, 0.5, 2.5, 5.0}.
4. De-embeds the perturbed vectors back into natural language with Vec2Text (batched for speed).
5. Prints a structured comparison of the reconstructed texts.
"""

import random
from typing import List
import torch
from sklearn.datasets import fetch_20newsgroups
import vec2text


def clean_text(text: str, max_chars: int = 250) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "..."
    return cleaned


def embed_texts(
    texts: List[str],
    corrector: vec2text.trainers.Corrector,
    device: str,
) -> torch.Tensor:
    inputs = corrector.embedder_tokenizer(
        texts,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    return embeddings.clone().detach()


def main():
    # Set seed for reproducible variable selection
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Fetch 20 Newsgroups dataset
    print("Loading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset="train",
        categories=["sci.space", "sci.crypt"],
        remove=("headers", "footers", "quotes"),
    )

    space_texts = []
    for text, target in zip(dataset.data, dataset.target):
        cleaned = clean_text(text)
        if len(cleaned) >= 100 and dataset.target_names[target] == "sci.space":
            space_texts.append(cleaned)

    # 2. Load Corrector
    print("Loading Vec2Text corrector for 'gtr-base'...")
    corrector = vec2text.load_pretrained_corrector("gtr-base")

    # 3. Embed a representative sample pool to compute coordinate standard deviations
    sample_pool = space_texts[:30]
    X_pool = embed_texts(sample_pool, corrector, device)
    dim = X_pool.shape[1]

    # Standard deviation per dimension across the dataset
    stds = X_pool.std(dim=0)

    # Choose an anchor text to perturb
    anchor_idx = 0
    anchor_text = space_texts[anchor_idx]
    anchor_vec = X_pool[anchor_idx : anchor_idx + 1]  # shape: (1, 768)

    # Baseline reconstruction (alpha = 0)
    with torch.no_grad():
        baseline_decoded = vec2text.invert_embeddings(
            embeddings=anchor_vec,
            corrector=corrector,
            num_steps=10,
        )[0]

    print("\n" + "=" * 80)
    print("ANCHOR TEXT (Original):")
    print(f"  {anchor_text}")
    print(f"\nBASELINE RECONSTRUCTION (alpha = 0):")
    print(f"  {baseline_decoded}")
    print("=" * 80)

    # 4. Randomly select 4 variables
    num_vars = 4
    selected_vars = sorted(random.sample(range(dim), num_vars))
    print(f"\nRandomly selected 4 variables (indices in [0, {dim-1}]): {selected_vars}")

    alpha_values = [-5.0, -2.5, -0.5, 0.5, 2.5, 5.0]

    # 5. Run sweep over variables and alpha values
    for var_idx in selected_vars:
        std_val = stds[var_idx].item()
        mean_val = X_pool[:, var_idx].mean().item()
        print("\n" + "#" * 80)
        print(f"VARIABLE (Dimension) #{var_idx}: Mean={mean_val:.4f}, Std={std_val:.4f}")
        print("#" * 80)

        # Build batched perturbed vectors for all alphas: shape (len(alpha_values), 768)
        perturbed_batch = anchor_vec.repeat(len(alpha_values), 1)
        deltas = []
        for i, alpha in enumerate(alpha_values):
            delta = alpha * (std_val if std_val > 0 else 0.03)
            perturbed_batch[i, var_idx] += delta
            deltas.append(delta)

        # Batch inversion
        with torch.no_grad():
            decoded_batch = vec2text.invert_embeddings(
                embeddings=perturbed_batch,
                corrector=corrector,
                num_steps=10,
            )

        for alpha, delta, decoded in zip(alpha_values, deltas, decoded_batch):
            print(f"  alpha = {alpha:+4.1f} (delta = {delta:+.4f}) -> \"{decoded}\"")

    print("\n" + "=" * 80)
    print("Sweep experiment finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
