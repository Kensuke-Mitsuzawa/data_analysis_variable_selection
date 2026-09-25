# Environment settings

Based on an inspection of this machine's hardware and environment profile:

- **GPU:** NVIDIA GeForce RTX 3060 (12 GB VRAM, CUDA 12.1 / PyTorch 2.5.1+cu121)
- **CPU / RAM:** 56 vCPUs (Intel Xeon E5-2680 v4) with 141 GB RAM (~105 GB free)
- **Disk:** 138 GB available
- **Python Environment:** Python 3.11 virtual environment at [`/root/data_analysis_variable_selection/.venv`](file:///root/data_analysis_variable_selection/.venv) managed via `uv`.

Here are the feasible model pairs of embedding and de-embedding models evaluated for this machine environment and the **20 Newsgroups** dataset (`sci.space` vs. `sci.crypt`).

---

### Comparison of Feasible Model Pairs

| Model Pair | Embedding Model (Encoder) | Latent Dim ($d$) | De-Embedding Model (Vec2Text Inverter) | Est. VRAM Footprint | Native `vec2text` API Support | Local Execution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1 (Recommended)** | `sentence-transformers/gtr-base` | 768 | `jxm/gtr__nq__32` + `jxm/gtr__nq__32__correct` | ~3.0 – 3.5 GB | **Yes** (`load_pretrained_corrector("gtr-base")`) | 100% Local |
| **Option 2 (Lightweight)** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | `vec2text/sbert-L6-128-noise-0.00001` | ~1.5 – 2.0 GB | Custom load via `load_corrector` | 100% Local |
| **Option 3 (High-Capacity)** | `thenlper/gte-base` | 768 | `vec2text/gte-512-noise-0.00001` | ~3.5 – 4.0 GB | Custom load via `load_corrector` | 100% Local |
| **Option 4 (API-Dependent)** | `text-embedding-ada-002` (OpenAI) | 1,536 | `jxm/vec2text__openai_ada002__msmarco__msl128__corrector` | ~3.0 GB (decoder only) | **Yes** (`load_pretrained_corrector("text-embedding-ada-002")`) | Requires Paid API |

---

### Detailed Evaluation

#### 1. Option 1: `GTR-Base` + `vec2text` (Recommended Choice)
* **Embedding Model:** `sentence-transformers/gtr-base` (T5-based dual-encoder for retrieval, 768 dimensions).
* **De-embedding Model:** Built into `vec2text.load_pretrained_corrector("gtr-base")`.
* **Hardware Suitability:**
  - The encoder (~440 MB) and decoder corrector (~1.5 GB) fit comfortably inside the RTX 3060’s 12 GB VRAM, leaving >8 GB for batch evaluation and MMD kernel matrices.
* **Why it fits the task:**
  - Direct 1-line loader support in `vec2text`:
    ```python
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    ```
  - `gtr-base` is pre-trained specifically on passage retrieval and query relevance, making it exceptionally sensitive to distinct topical boundaries like `sci.space` (astronomy, NASA, orbit) vs. `sci.crypt` (encryption, PGP, NSA, keys).

---

#### 2. Option 2: `SBERT` (`all-MiniLM-L6-v2`) + `vec2text/sbert-L6-128` (Fast / Low-Resource Alternative)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
* **De-embedding Model:** `vec2text/sbert-L6-128-noise-0.00001`.
* **Hardware Suitability:**
  - Minimal VRAM footprint (<2 GB).
* **Why it fits the task:**
  - Lower latent dimensionality ($d = 384$ vs $768$) speeds up the MMD variable selection and pairwise distance matrices significantly.
* **Trade-off:**
  - Requires loading via the lower-level `vec2text.load_corrector()` rather than the 1-line helper `load_pretrained_corrector()`.

---

#### 3. Option 4: `text-embedding-ada-002` (Not Recommended for Local Testing)
* **Why to avoid for this pilot:**
  - Although `vec2text` officially supports `load_pretrained_corrector("text-embedding-ada-002")`, generating the embeddings requires an OpenAI API key, network latency, and continuous API costs.
  - Furthermore, $d=1536$ is twice as wide as GTR-base, increasing the search space for MMD-CV without offering local determinism.

---

### Proposed Setup for Pilot Project

For the pilot script in [`/root/data_analysis_variable_selection/plans/plan_2026-09-25-1536_pilot_project_vec2text_project`](file:///root/data_analysis_variable_selection/plans/plan_2026-09-25-1536_pilot_project_vec2text_project):

1. **Install dependencies** into the active venv:
   ```bash
   uv pip install --python /root/data_analysis_variable_selection/.venv/bin/python vec2text sentence-transformers
   ```
2. **Standard Workflow with GTR-Base**:
   - Fetch 20 Newsgroups (`sci.space` vs `sci.crypt`) via `sklearn.datasets.fetch_20newsgroups`.
   - Embed text samples using `sentence-transformers/gtr-base` ($X, Y \in \mathbb{R}^{n \times 768}$).
   - Load the decoder using `vec2text.load_pretrained_corrector("gtr-base")`.
   - Invert selected vector representations directly with `vec2text.invert_embeddings()`.

---

## Execution Log & Pilot Verification

### 1. Package Installation
Dependencies installed via `uv add`:
- `vec2text` (v0.0.13)
- `sentence-transformers` (v5.1.2)
- `transformers` pinned to `<4.48.0` (`4.47.1`) to ensure full compatibility with PyTorch 2.5.1 and pre-trained `.bin` weights.
- `datasets` updated to `>=2.14.0` (`5.0.1`) to ensure compatibility with `pyarrow>=25`.

### 2. Pilot Script
The executable script is located at:
[`/root/data_analysis_variable_selection/plans/plan_2026-09-25-1536_pilot_project_vec2text_project/run_pilot_vec2text.py`](file:///root/data_analysis_variable_selection/plans/plan_2026-09-25-1536_pilot_project_vec2text_project/run_pilot_vec2text.py)

### 3. Execution Results
Ran with GPU acceleration (`device: cuda` on NVIDIA RTX 3060):
- **Dataset Loaded:** 20 Newsgroups (`sci.space` vs. `sci.crypt`).
- **Embedding Dimensions:** $X, Y \in \mathbb{R}^{n \times 768}$.
- **Reconstruction Demonstrations:**
  - **Distribution X (`sci.space`):**
    - *Original:* `"Any prize like this is going to need to be worded carefully enough that you cannot get it without demonstrating sustained and reliable capability..."`
    - *Decoded:* `"one can get this prize with a word of one, and not a very reliable and sustained capability. This will usually have to be done carefully."`
  - **Distribution Y (`sci.crypt`):**
    - *Original:* `"Hi ! I am interested in the source of FEAL encryption algorithm. Does someone of you know where I can get the source from, or where I can find documentation about FEAL..."`
    - *Decoded:* `"I want to know about the source of FEAL encryption. I have already done some documentation on this source and I have some documentation on Hermann Rau"`
  - **Latent Coordinate Perturbation (Variable Selection Demonstration):**
    - Isolated latent dimension `#654` and shifted it by $+0.04$.
    - *Baseline Reconstruction:* `"one can get this prize with a word of one, and not a very reliable and sustained capability. This will usually have to be done carefully."`
    - *Perturbed Reconstruction:* `"you can get this prize with a wordy one with sustained and reliable capability. It must be done with very little skill and not a shot."`

---

## Technical Q&A: Reconstruction Fidelity & Determinism

### 1. Is it normal that `vec2text` does not reconstruct the exact text?
**Yes, entirely normal.**
- Dense embeddings compress variable-length text (tens or hundreds of words) into a fixed-dimensional vector ($d=768$).
- This is an **information-lossy, many-to-one projection**. Multiple semantically similar phrases map to nearly identical points on the unit sphere.
- As reported in the original paper (*Morris et al., ICLR 2024*):
  - Short sentences ($\le 32$ tokens) achieve high exact match ($\sim 92\%$ with 50+ recursive steps and beam search).
  - Longer passages (like 20 Newsgroups posts) have lower exact token match, but achieve high **semantic paraphrase fidelity** (preserving key nouns, verbs, domain semantics, and topical themes).

### 2. Is de-embedding stochastic or deterministic?
**It is deterministic at evaluation time.**
- In `vec2text`, generation uses `do_sample=False` (greedy / beam search decoding).
- Training noise (`noise_level * torch.randn`) is strictly disabled in `eval()` mode.
- Running `invert_embeddings()` multiple times on the exact same input tensor yields bitwise identical decoded strings.

---

## Variable Perturbation Sweep ($\alpha \in \{-5.0, -2.5, -0.5, 0.5, 2.5, 5.0\}$)

Script: [`/root/data_analysis_variable_selection/plans/plan_2026-09-25-1536_pilot_project_vec2text_project/run_variable_alpha_sweep.py`](file:///root/data_analysis_variable_selection/plans/plan_2026-09-25-1536_pilot_project_vec2text_project/run_variable_alpha_sweep.py)

Tested on anchor text from `sci.space`:
> *"Any prize like this is going to need to be worded carefully enough that you cannot get it without demonstrating sustained and reliable capability, rather than a lucky one-shot. It can be done."*

Baseline Reconstruction ($\alpha = 0$):
> *"one can get this prize with a word of one, and not a very reliable and sustained capability. This will usually have to be done carefully."*

### Results for 4 Random Variables:
- **Variable #25:**
  - $\alpha = -5.0$ -> `"It is usually not easy to accomplish this with a prize that is only "shot with one word and a sustained and reliable power". However, it"`
  - $\alpha = -2.5$ -> `"It is a simple way to get this prize if it is done with a one word, sustained and reliable 'shot'. Unfortunately,"`
  - $\alpha = -0.5$ -> `"one can get this prize with a "word that is reliably and not sustained power". This is a very difficult task, since it would have"`
  - $\alpha = +0.5$ -> `"It is unlikely that you can get this prize with one worded and sustained "reliable" capability. It is easy to do, but it must"`
  - $\alpha = +2.5$ -> `"One way this prize can be achieved is to get it with a one-worded, not very sustained and reliable operation. It is usually thought"`
  - $\alpha = +5.0$ -> `"You can get this prize with a simple one worded way, requiring careful resonance and sustained capability to be dealt with. However"`

- **Variable #114:**
  - $\alpha = -5.0$ -> `"One can get this prize if it is done with a simple wording and sustained "reliableness" that does not require a hit."`
  - $\alpha = -2.5$ -> `"one can get this prize if it is done with a simple "word with sustained and reliable power" and not a particular skillful handling."`
  - $\alpha = -0.5$ -> `"It is not usually done that you can get a prize with one worded and reliable sustained capability. This is a shot that will be difficult"`
  - $\alpha = +0.5$ -> `"It is usually not easy to get this prize with one wording and a limited chance of reliably sustained and capableness. It may have to be"`
  - $\alpha = +2.5$ -> `"It is not easy to get this prize with a word-based one that is reliably sustained and capable. However, this will usually have to be"`
  - $\alpha = +5.0$ -> `"It is a very simple trick to get this prize with one word and not with sustained-reliable aplombs. It would have to"`

- **Variable #654:**
  - $\alpha = -5.0$ -> `"It is usually possible to get this prize with only one word, unless one is capable of sustained and reliable resonance. It isn'"`
  - $\alpha = -2.5$ -> `"It is not usually done that one can get this prize with a simple wording that requires sustained and reliably power. This is a rather tricky"`
  - $\alpha = -0.5$ -> `"one must do this with one worded and a shot that is not sustained and capable. The chances of getting the prize can be quite simple."`
  - $\alpha = +0.5$ -> `"it is unlikely that this prize can be won with a simple one worded "shot" and with sustained reliability. It is usually not done well"`
  - $\alpha = +2.5$ -> `"This prize can be done with a very simple one worded "shot" with sustained capability and reliability. It is unlikely to ever get it out"`
  - $\alpha = +5.0$ -> `"It is not easy to do this with one word, but a prize that can be "shot with sustained and reliable capability" if it is given"`

- **Variable #759:**
  - $\alpha = -5.0$ -> `"It is unlikely that you can get this prize with a simple one worded and reliable shot with sustained capability. It will be easy to do that"`
  - $\alpha = -2.5$ -> `"You can get this prize by simply doing it with a one-shot and very reliable sustained capabilities. It will not be a word of mouth that"`
  - $\alpha = -0.5$ -> `"One way to get this prize is to do it is with a word and not with any sustained reliably capability, and one will usually be shot out"`
  - $\alpha = +0.5$ -> `"One way to get this prize is to do it with a one-shot with very limited wording and reliable sustained capability, and it will then be"`
  - $\alpha = +2.5$ -> `"one worded version of this prize is usually done with a shot and with sustained and reliably capability. It will not be easy to get it"`
  - $\alpha = +5.0$ -> `"One way this prize can be done is to get it with a word-only "reliable and sustained capability" that may not be very straightforward."`