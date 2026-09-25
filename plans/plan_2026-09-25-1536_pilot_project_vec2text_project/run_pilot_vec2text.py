"""Pilot script to demonstrate embedding and de-embedding using GTR-Base and Vec2Text.

Tasks:
1. Fetch 20 Newsgroups (sci.space vs sci.crypt) via sklearn.datasets.fetch_20newsgroups.
2. Embed text samples using sentence-transformers/gtr-base (X, Y in R^{n x 768}).
3. Load the decoder using vec2text.load_pretrained_corrector("gtr-base").
4. Invert selected vector representations directly with vec2text.invert_embeddings().
5. Randomly select a latent variable in the dense vector space and unpack/reconstruct
   its semantic effect in the text space.
"""

import random
from typing import List
import torch
from sklearn.datasets import fetch_20newsgroups
import vec2text


def clean_text(text: str, max_chars: int = 250) -> str:
    """Clean whitespace and truncate text for clear comparison."""
    cleaned = " ".join(text.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + "..."
    return cleaned


def embed_texts(
    texts: List[str],
    corrector: vec2text.trainers.Corrector,
    device: str,
) -> torch.Tensor:
    """Embed texts using the exact paired GTR-base encoder."""
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Vec2Text Pilot: GTR-Base Embedding & Inversion")
    print(f"Device: {device}")
    print(f"==================================================")

    # 1. Fetch 20 Newsgroups dataset (sci.space vs sci.crypt)
    categories = ["sci.space", "sci.crypt"]
    print(f"\n[1/4] Fetching 20 Newsgroups dataset ({categories[0]} vs {categories[1]})...")
    dataset = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    space_texts: List[str] = []
    crypt_texts: List[str] = []

    for text, target in zip(dataset.data, dataset.target):
        cleaned = clean_text(text)
        # Filter for meaningful paragraphs
        if len(cleaned) >= 80:
            if dataset.target_names[target] == "sci.space":
                space_texts.append(cleaned)
            else:
                crypt_texts.append(cleaned)

    # Take a sample subset for pilot demonstration
    sample_size = 15
    space_samples = space_texts[:sample_size]
    crypt_samples = crypt_texts[:sample_size]
    print(f"Retrieved {len(space_samples)} 'sci.space' and {len(crypt_samples)} 'sci.crypt' samples.")

    # 2. Load Vec2Text corrector (paired GTR-base encoder + decoder)
    print("\n[2/4] Loading Vec2Text corrector for 'gtr-base'...")
    corrector = vec2text.load_pretrained_corrector("gtr-base")

    # 3. Embed text samples using the paired GTR-base encoder (X, Y in R^{n x 768})
    print("\n[3/4] Embedding text samples into dense vector space (d=768)...")
    X = embed_texts(space_samples, corrector, device)
    Y = embed_texts(crypt_samples, corrector, device)

    print(f"  Distribution X (sci.space) shape: {X.shape}")
    print(f"  Distribution Y (sci.crypt) shape: {Y.shape}")

    # 4. Invert selected vector representations directly with vec2text.invert_embeddings()
    print("\n[4/4] Inverting vector representations with vec2text.invert_embeddings()...")

    # (a) Reconstruct a sample vector from Distribution X (sci.space)
    idx_x = 0
    target_vec_x = X[idx_x : idx_x + 1]
    with torch.no_grad():
        decoded_x = vec2text.invert_embeddings(
            embeddings=target_vec_x,
            corrector=corrector,
            num_steps=10,
        )

    print("\n" + "=" * 60)
    print("DEMO A: Decoding Sample from Distribution X ('sci.space')")
    print("=" * 60)
    print(f"Original Text:\n  {space_samples[idx_x]}")
    print(f"\nDecoded Text:\n  {decoded_x[0]}")

    # (b) Reconstruct a sample vector from Distribution Y ('sci.crypt')
    idx_y = 0
    target_vec_y = Y[idx_y : idx_y + 1]
    with torch.no_grad():
        decoded_y = vec2text.invert_embeddings(
            embeddings=target_vec_y,
            corrector=corrector,
            num_steps=10,
        )

    print("\n" + "=" * 60)
    print("DEMO B: Decoding Sample from Distribution Y ('sci.crypt')")
    print("=" * 60)
    print(f"Original Text:\n  {crypt_samples[idx_y]}")
    print(f"\nDecoded Text:\n  {decoded_y[0]}")

    # (c) Dense variable selection & reconstruction:
    # Select a variable (dimension) in the dense vector space and unpack its semantic meaning in text space
    print("\n" + "=" * 60)
    print("DEMO C: Dense Variable Selection & Semantic Reconstruction")
    print("=" * 60)

    # Pick a random latent variable index j in {0, ..., 767}
    selected_var_idx = random.randint(0, X.shape[1] - 1)
    print(f"Selected Latent Dimension (Variable): #{selected_var_idx}")

    # Measure the shift between Distribution X and Y along this specific variable
    mean_diff = (X[:, selected_var_idx].mean() - Y[:, selected_var_idx].mean()).item()
    std_x = X[:, selected_var_idx].std().item()
    print(f"Variable #{selected_var_idx} Stats:")
    print(f"  Mean in X (sci.space): {X[:, selected_var_idx].mean().item():.4f}")
    print(f"  Mean in Y (sci.crypt): {Y[:, selected_var_idx].mean().item():.4f}")
    print(f"  Delta (X - Y):         {mean_diff:+.4f}")

    # Perturb the anchor vector along the selected variable
    perturbed_vec = target_vec_x.clone()
    perturbation_scale = 5.0 * (std_x if std_x > 0 else 1.0)
    perturbed_vec[0, selected_var_idx] += perturbation_scale

    with torch.no_grad():
        decoded_perturbed = vec2text.invert_embeddings(
            embeddings=perturbed_vec,
            corrector=corrector,
            num_steps=10,
        )

    print(f"\nBaseline Reconstruction (Unperturbed):\n  {decoded_x[0]}")
    print(f"\nReconstruction after Shifting Variable #{selected_var_idx} by +{perturbation_scale:.2f}:\n  {decoded_perturbed[0]}")
    print("\n" + "=" * 60)
    print("Pilot demonstration finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
