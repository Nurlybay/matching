# Product Card Matching — GBM Pipeline

A solution to the candidate product-pair classification task (ecup26-matching).
The pairs are pre-assembled, so there is no retrieval part to solve — for every
pair $(i, j)$ the task is to output $\hat{p}_{ij} \in [0, 1]$, the probability
that the two cards describe the same product.

The competition metric is **PR-AUC computed separately for each of the 20
categories and averaged** (§5.3).

| Solution | Validation | Leaderboard |
|---|---:|---:|
| String features only | 0.5565 | 0.2650 |
| + MiniLM-L12, 2M-pair pretraining | 0.7060 | 0.3626 |
| + MiniLM-L12, 11.19M-pair pretraining | 0.7417 | 0.4245 |
| **+ e5-base, 11.19M-pair pretraining** | **0.7827** | **0.4592** |

## Architecture

```
pair (i, j)
   ├─ 19 features: name string similarity, numbers, transliteration,
   │  attributes, identifiers, category code                       (§2)
   └─ cross-encoder logit                                          (§7)
         ├─ pretraining on 11.19M LLM-labelled pairs, soft labels
         └─ fine-tuning on 292K human-labelled pairs
                                    ↓
              HistGradientBoostingClassifier               (§4)
                                    ↓
                        probability of a duplicate
```

Four findings shaped the solution — all of them came from measurement, and
three of them refuted the premises the pipeline started from:

1. **The inference budget is four times larger than needed** (§7.1). The
   starting premise was "speed is a hard constraint, so cheap features only, no
   neural network." Measurement showed ~170–195 s against a 780 s limit, with
   the GPU completely idle. Giving up the neural network cost about $0.13$
   ROC-AUC and bought nothing.
2. **LLM labels are valuable, but not as training rows** (§6.2). Mixed into the
   human labels they *hurt* quality — monotonically, all the way to the
   degenerate case of "no LLM data at all." Moved into a separate pretraining
   stage, they produce the largest gain in the entire project.
3. **We optimized the wrong metric** (§5.3). The first twenty decisions were
   made on pooled ROC-AUC, while the evaluation is a macro-average of PR-AUC
   over categories. These are different quantities: the pooled version weights
   a category by its number of pairs, the macro version puts all twenty on
   equal footing.
4. **Validation understates the value of pretraining** (§6.3). Switching to
   11M-pair pretraining gave +5.5% on validation and +17% on the leaderboard.
   Validation consists of products from the same narrow subset that fine-tuning
   uses, so it cannot see the main effect — the ability to generalize to unseen
   products.

---

## 1. Problem statement

Let $x_i = (\text{name}_i, \text{attributes}_i, \text{category}_i)$ be a product
card. The task is to build

$$f: (x_i, x_j) \mapsto \hat{p}_{ij} \in [0,1].$$

The training set is labelled from two sources of fundamentally different nature:

| Source | Size | Nature of $t$ | Meaning |
|---|---:|---|---|
| `matches.parquet` | 365,654 | $t \in \{0, 1\}$ | human labels, ground truth |
| `matches_llm.parquet` | 11,187,780 | $t \in [0, 1]$ | probabilistic LLM labels |

Catalogue: `items.parquet` — 13,397,761 products across 20 categories;
`items_human.parquet` — 711,304 products (the subset covering the human-labelled
pairs).

The $\approx 30{:}1$ asymmetry in favour of the LLM labels is the central fact
around which the entire validation methodology below is built (§5).

---

## 2. Features

17 features per pair, all built from the Python standard library, with no
external dependencies. Notation: $s_1, s_2$ are the name strings,
$T_1, T_2$ the token sets of the names, $A_1, A_2$ the attribute dictionaries,
$K_1 = \operatorname{dom} A_1$, $K_2 = \operatorname{dom} A_2$ the attribute key
sets, and $V_1, V_2$ the token sets of the attribute values.

Tokenization: `[a-zA-Zа-яА-ЯёЁ0-9]+` after lowercasing.

### 2.1 Name features

**`name_seq_ratio`** — the Ratcliff–Obershelp coefficient:

$$\text{ratio}(s_1, s_2) = \frac{2M}{|s_1| + |s_2|},$$

where $M$ is the total length of all matching blocks found by the recursive
longest-common-substring algorithm (`difflib.SequenceMatcher`). Unlike
Levenshtein distance, it accounts for character order and is robust to
reordering of large blocks — which matters for product names of the form
`"brand sku type"` vs `"type brand sku"`.

Complexity is $O(n^2)$ in the worst case; this is the most expensive feature in
the set, and it is what determines the pipeline's throughput (≈11K pairs/s
single-threaded).

**`name_token_jaccard`** — the Jaccard index over tokens:

$$J(T_1, T_2) = \frac{|T_1 \cap T_2|}{|T_1 \cup T_2|}.$$

It complements the previous feature: invariant to word order, but blind to typos
and morphology. The pair $(\text{ratio}, J)$ separates two different kinds of
divergence — "same words in a different order" ($J$ high, ratio lower) and
"similar words with typos" (the reverse).

**`name_len_ratio`**, **`name_len_diff`**:

$$\frac{\min(|s_1|,|s_2|)}{\max(|s_1|,|s_2|,1)}, \qquad \bigl||s_1| - |s_2|\bigr|.$$

A proxy for "one card is a verbose description, the other is terse" — a typical
pattern for partial duplicates.

**`common_word_count`** — $|T_1 \cap T_2|$, an unnormalized counter. Unlike $J$,
it preserves absolute scale: 8 shared words out of 10 and 2 out of 2 carry
different signal even though the latter has a higher $J$.

### 2.2 Attribute features

**`attr_key_jaccard`** — $J(K_1, K_2)$, how comparable the attribute schemas are.

**`attr_common_key_count`** — $|K_1 \cap K_2|$.

**`attr_exact_match_ratio`** — the share of matching values among shared keys:

$$\frac{\bigl|\{k \in K_1 \cap K_2 : A_1(k) = A_2(k)\}\bigr|}{|K_1 \cap K_2|}.$$

The most direct match signal: agreement on `артикул` (SKU), `партномер` (part
number) or `oem-номер` (OEM number) almost deterministically means a duplicate.

**`attr_value_jaccard`** — $J(V_1, V_2)$, robust to disagreement in key naming
(when one seller writes `бренд` and another writes `производитель`).

**`attr_count_diff`** — $\bigl||A_1| - |A_2|\bigr|$, how completely the card is
filled in.

**`category_match`** — $\mathbb{1}[c_1 = c_2]$.

### 2.3 Features found by error analysis

Four features were added not from general reasoning but after reading concrete
validation errors (`src/analyze.py`).

**`category_code`** — the category code of the pair. A category *consistency*
feature is useless (candidates are always within one category, so
`category_match` is identically zero), but the measured per-category ROC-AUC
ranges from $0.67$ (Auto parts) to $0.95$ (Pharmacy) — the model is effectively
solving twenty different tasks without knowing which one it faces. The feature
is declared categorical so the tree splits on arbitrary subsets of categories
rather than on meaningless thresholds like "code < 7."

**`number_jaccard`, `number_count_diff`** — agreement of numeric tokens in the
names. Nearly every error examined came down to numbers: $-2.00$ vs $-3.00$
dioptres, $300{\times}220$ vs $220$ curtains, $66\text{-}68$ vs $68\text{-}70$
pupillary distance. Token features drown such differences out: for a pair of
glasses differing by one dioptre, $J(T_1,T_2) = 0.636$, while over numbers it is
$0.333$.

**`name_translit_ratio`** — name similarity after folding Cyrillic onto Latin.
The pair "кроссовки asics" / "кроссовки асикс" is labelled a duplicate, yet
token Jaccard gives $0.333$ since there are no shared tokens at all; after
transliteration it is $0.933$. Latin and Cyrillic spellings of brands coexist
throughout the catalogue.

**`id_attr_match`** — agreement on identifying attributes (`артикул`,
`партномер`, `oem-номер`, `модель`), normalized to alphanumeric form so that
`8500892sx` and `8500892-SX` match. NaN if either side has no identifier — which
is the majority of cards, and substituting $0$ would falsely assert "we checked,
they differ." The feature targets Auto parts: when the cross-encoder was added,
that category gained only $+0.005$ against $+0.128$ for Beauty & Hygiene — that
is what subword tokenization looks like when it shreds alphanumeric codes.

### 2.4 The semantics of missing values

A separate substantive question is what to return when a feature is
**undefined**. The naive implementation returned $J(\varnothing, \varnothing) = 1$
("two empty sets match perfectly"), while `attr_exact_match_ratio` returned $0$
in the analogous situation. In other words, absence of information was encoded
sometimes as the top of the scale and sometimes as the bottom — and in the first
case the model was handed a "perfect match" where there was nothing to compare.
For cards with empty attributes that is a direct generator of false positives.

The current convention:

$$J(A, B) = \begin{cases}
\text{NaN}, & A = B = \varnothing \quad \text{(no information)} \\
0, & A = \varnothing \veebar B = \varnothing \quad \text{(a real difference)} \\
\dfrac{|A \cap B|}{|A \cup B|}, & \text{otherwise}
\end{cases}$$

`HistGradientBoostingClassifier` handles NaN natively: at each split, missing
values are routed to whichever branch yields the larger gain, so the optimal
direction is **learned from the data** rather than set by hand.

Two additional indicators were introduced, `attrs1_empty` and `attrs2_empty` —
$\mathbb{1}[|A| = 0]$. An empty card becomes an observable feature instead of
masquerading as a similarity value.

---

## 3. Working with two label sources

Binary label: $y = \mathbb{1}[t \geq 0.5]$.

Sample weights:

$$w_i = \begin{cases}
\lambda_H, & i \in \text{human} \\
2\,|t_i - 0.5|, & i \in \text{LLM}
\end{cases}$$

The factor $2|t - 0.5|$ maps $[0,1] \to [0,1]$ with a minimum at $t = 0.5$:
confident LLM judgements ($t \to 0$ or $t \to 1$) get full weight, while
uncertain ones get a weight approaching zero. In essence this is a linear
approximation of the labelling classifier's confidence.

Pairs with $t = 0.5$ are **dropped**, not merely given $w = 0$: the rule
$y = \mathbb{1}[t \geq 0.5]$ would mark them positive, and they would remain in
the matrix as rows with a label backed by no signal at all. Zero weight excludes
them from the loss, but not from split statistics nor from the
`validation_fraction` inside the boosting itself.

$\lambda_H$ is not chosen by intuition but searched over the grid
$\{1, 3, 5, 10\}$; the value with the best ROC-AUC on the honest validation set
wins (§5).

---

## 4. Model

`HistGradientBoostingClassifier` — histogram-based gradient boosting (a
LightGBM-style implementation). It minimizes the weighted logistic loss:

$$\mathcal{L} = -\sum_i w_i \bigl[y_i \log p_i + (1 - y_i)\log(1 - p_i)\bigr],
\qquad p_i = \sigma(F(x_i)).$$

The ensemble is built additively: $F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$, where
$\nu$ is the learning rate and each new tree $h_m$ approximates the Newton step.
For the logistic loss, the per-sample gradient and Hessian are:

$$g_i = w_i\,(p_i - y_i), \qquad h_i = w_i\,p_i(1 - p_i).$$

The optimal value in leaf $L$ under L2 regularization $\lambda$:

$$v_L^{*} = -\frac{\sum_{i \in L} g_i}{\sum_{i \in L} h_i + \lambda},$$

and the gain from splitting a node into $L$ and $R$:

$$\text{Gain} = \frac{1}{2}\left[
\frac{(\sum_{L} g)^2}{\sum_{L} h + \lambda} +
\frac{(\sum_{R} g)^2}{\sum_{R} h + \lambda} -
\frac{(\sum_{L \cup R} g)^2}{\sum_{L \cup R} h + \lambda}
\right].$$

This is where the weighting mechanism becomes visible: $w_i$ enters both $g_i$
and $h_i$, so it scales a sample's contribution to the choice of split, not just
to the final loss.

**Why the histogram variant.** Features are pre-quantized into 255 bins, after
which finding the optimal split on a feature costs $O(\#\text{bins})$ instead of
$O(n \log n)$ for sorting. At $n \approx 1.15 \times 10^7$ that is the
difference between hours and minutes. The histogram-subtraction trick applies on
top: a child node's histogram is obtained by subtracting its sibling's from the
parent's, saving a pass over the data.

Hyperparameters: `max_iter=500`, `learning_rate=0.05`, `max_depth=8`,
`l2_regularization=1.0`, early stopping with `n_iter_no_change=20`.

---

## 5. Validation methodology

This is the most substantive part of the solution — two mistakes were fixed
here, each of which produced an inflated and therefore useless metric.

### 5.1 Validate on human labels only

Initially, human and LLM features were concatenated and then split with a random
`train_test_split`. At a $30{:}1$ ratio, the validation set ended up
$\approx 97\%$ LLM-labelled. The metric formally went up, but what it measured
was **agreement with the LLM labeller**, not with the truth — while the test set
is human-labelled.

The correct order: the hold-out is carved out of the human labels **before** any
mixing, LLM pairs go exclusively into train, and the metric is computed only on
the held-out human portion.

### 5.2 Leakage through connected components

Pairs are not independent observations: they are linked through shared products.
If $(X, A)$ lands in train and $(X, B)$ in validation, the model sees on
validation a card it has already studied. In product matching this is aggravated
by clusters of near-identical cards, which produce dozens of nearly coincident
feature vectors.

Formally: consider the graph $G = (V, E)$ where $V$ are products and $E$ are
labelled pairs. The split must cut not edges but **connected components**:

$$V = \bigsqcup_k C_k, \qquad
\text{train} = \bigcup_{k \in S} E[C_k], \quad
\text{val} = \bigcup_{k \notin S} E[C_k].$$

Components are built with a disjoint-set (union-find) structure using path
compression and union by rank — amortized complexity
$O(|E|\,\alpha(|V|))$, where $\alpha$ is the inverse Ackermann function
(effectively constant). The split itself is performed by `GroupShuffleSplit`
with `groups = component_id`.

On the human labels: **345,654 components over 365,654 pairs** — that is, the
graph breaks into many small clusters rather than one giant one. As a result the
component-wise split stays close to 80/20 (292,370 / 73,284 in practice).

The same leak was also closed at the boundary between sources: any LLM-train
pair touching a product from the held-out human validation set is excluded.

### 5.3 The metric: macro-averaged PR-AUC

The competition computes PR-AUC separately for each of the 20 categories and
averages them:

$$\text{score} = \frac{1}{20}\sum_{c=1}^{20} \text{PR-AUC}\bigl(y^{(c)}, \hat{p}^{(c)}\bigr).$$

The first twenty decisions in this project were made on pooled ROC-AUC, which is
wrong along two axes at once.

**Pooled vs macro-average.** In the pooled metric a category's weight is
proportional to its number of pairs; in the macro-average all twenty are equal.
A measured pooled PR-AUC of $0.6388$ corresponded to a macro value of $0.5648$.
The practical consequence: improving the weakest category is worth exactly as
much as improving the strongest — and there is far more headroom in the weak
ones.

**ROC-AUC vs PR-AUC.** ROC-AUC $= P(\hat{p}_+ > \hat{p}_-)$ is invariant to the
positive rate and only weakly sensitive to how many negatives sit at the very
top of the ranking. PR-AUC measures mostly exactly that.

**The cross-sample comparison trap.** The lower bound of PR-AUC equals the
positive rate, so values are not comparable across sets with different class
balance. In our validation set the rate is $0.257$, and categories with rare
duplicates yield a low PR-AUC even with excellent ranking:

| Category | Positive rate | PR-AUC | ROC-AUC | Lift over baseline |
|---|---:|---:|---:|---:|
| Shoes | 0.087 | 0.443 | 0.872 | **5.07×** |
| Hobbies & Crafts | 0.471 | 0.883 | 0.907 | 1.88× |

Shoes ranks five times better than chance and contributes half as much to the
metric as Hobbies, which ranks twice as well as chance. In such categories what
needs improving is not overall quality but precision at the top of the list.

Feature importances are computed via **permutation importance** on the held-out
human validation set rather than via built-in impurity-based importances — the
latter systematically overstate the contribution of high-cardinality features.

---

## 6. Results

All numbers are on the same group-aware split, with only human labels in the
hold-out.

### 6.1 Overall trajectory

| Configuration | ROC-AUC | pooled PR-AUC | macro PR-AUC |
|---|---:|---:|---:|
| GBM, string features | 0.7496 | 0.5273 | — |
| + MiniLM-L12 (no pretraining) | 0.8282 | 0.6388 | 0.5648 |
| + category, numbers, transliteration, identifiers | 0.8468 | 0.6721 | — |
| + MiniLM-L12, 2M pretraining | 0.8961 | 0.7660 | 0.7060 |
| + MiniLM-L12, 11.19M pretraining | 0.9114 | 0.8008 | 0.7417 |
| **+ e5-base, 11.19M pretraining** | **0.9288** | **0.8367** | **0.7827** |

Two levers produced almost all of the gain, and both concern the cross-encoder:

| Lever | On validation | On the leaderboard |
|---|---:|---:|
| Pretraining volume: 2M → 11.19M | +0.036 | +0.062 |
| Model size: 118M → 278M | +0.041 | +0.035 |

The string features were devalued in the process: on top of a strong CE, the
whole GBM superstructure adds $0.7809 \to 0.7827$, i.e. $+0.0018$, while taking
26% of inference time (§6.4).

### 6.2 How to use the LLM labels

This is the main substantive finding of the project, and it took two attempts.

**What does not work: LLM pairs as training rows.** Sweeping the human-label
weight $\lambda_H$ while training the GBM on the union of both sources:

| $\lambda_H$ | ROC-AUC | PR-AUC |
|---|---:|---:|
| 1 | 0.7119 | 0.4780 |
| 3 | 0.7180 | 0.4867 |
| 5 | 0.7218 | 0.4922 |
| 10 | 0.7289 | 0.5017 |
| $\infty$ (= `--skip_llm`) | **0.7496** | **0.5273** |

The metric rises monotonically as the LLM rows are suppressed and peaks where
they have no influence at all. That is, $\lambda_H$ is not finding an optimum
inside the range but interpolating toward the edge — extending the grid upward
would be pointless.

**What does work: LLM labels as a separate pretraining stage.** The
cross-encoder is trained on the LLM pairs with soft labels, then fine-tuned on
the human labels. Soft labels, not binarized ones: `BCEWithLogitsLoss` is
minimized at $\sigma(z) = t$ for any $t \in [0,1]$, so a maximally uncertain
$t = 0.5$ by itself produces almost no gradient — uncertainty is handled by the
loss function rather than by external weighting.

Pretraining volume turned out to be the main lever of the whole project:

| Pairs in pretraining | ROC-AUC after pretraining, **with zero human labels** | Final ROC-AUC |
|---:|---:|---:|
| no pretraining | — | 0.8077 |
| 1M | 0.7114 | 0.8720 |
| 2M | 0.7959 | 0.8957 |
| **11.19M** (everything available) | **0.8548** | **0.9114** |

The gain decays — roughly $+0.085$ for the first doubling and $+0.024$ per
doubling afterwards — but it does not reach zero: the lever was closed not by
saturation but by running out of data.

**Why the two results do not contradict each other.** The key is the 0.7114 row:
a model that has never seen a single human-labelled pair already nearly catches
up with a GBM trained specifically on them (0.7496). So there is plenty of
transferable signal in the LLM labels, and what ruined them was not their own
noise but the **way they were mixed in**: 11M noisy rows and 365K clean ones in
a single loss at a $30{:}1$ ratio — the clean ones simply drown. As a separate
stage, on top of which fine-tuning rewrites the decision layer, the very same
labels work.

Indirect confirmation comes from the epoch dynamics. Without pretraining the
model hits a ceiling on the second epoch (0.8077) and declines on the third
(0.8074). With pretraining it is still improving at that same second epoch: a
pretrained representation lets it extract noticeably more from the same 292K
human-labelled pairs.

### 6.3 Calibrating validation against the leaderboard

The first submission scored $0.3626$ against a validation macro of $0.7060$ — a
twofold drop that the change of metric did not explain. To find the cause, one
submission was spent on **diagnostics**: the same solution without the
cross-encoder.

| Solution | Validation | Leaderboard | Transfer ratio |
|---|---:|---:|---:|
| String features only | 0.5565 | 0.2650 | 0.476 |
| MiniLM, 2M pretraining | 0.7060 | 0.3626 | 0.514 |
| MiniLM, 11.19M pretraining | 0.7417 | 0.4245 | 0.570 |
| e5-base, 11.19M pretraining | 0.7827 | **0.4592** | **0.587** |

Two conclusions.

**The gap is a scale factor, not a breakage.** Test numbers follow from
validation numbers by an approximately constant multiplier. The most likely
cause is a lower positive rate in the test set: the lower bound of PR-AUC equals
the positive rate, and with an average lift of $3.44\times$ over baseline on our
validation set, a result of $0.3626$ corresponds to a positive rate of about
$0.105$ against our $0.257$. Validation predicts the ordering of models
correctly; the absolute values live on their own scale.

**Validation understates every cross-encoder improvement.** The transfer ratio
grows monotonically: $0.476 \to 0.514 \to 0.570 \to 0.587$. And the relative
gain on the leaderboard beats the validation gain every time:

| Step | On validation | On the leaderboard |
|---|---:|---:|
| 2M → 11M pretraining | +5.5% | **+17%** |
| MiniLM → e5-base | +5.5% | **+8.2%** |

The explanation lies in how validation is constructed. It is assembled from
`items_human` products — the same narrow subset (711K out of 13.4M) that
fine-tuning runs on. Pretraining, however, covers the whole catalogue, and a
larger model generalizes better; both effects show up precisely on **unfamiliar**
products, which our validation set contains none of and the test set consists of
entirely.

The practical consequence for decision-making: a modest validation gain from
strengthening the cross-encoder should be read as a substantial gain on the test
set. This is a direction to push on, not to be cautious about.

### 6.4 Composition

The gain from stacking the GBM on top of the cross-encoder is asymmetric: with a
weak CE, ROC-AUC improved by $+0.027$ while PR-AUC improved by $+0.060$ — twice
as much in relative terms. At a positive rate of $0.257$ it is PR-AUC that
reflects practical usefulness.

As the CE gets stronger, the superstructure's contribution predictably shrinks:
on top of CE v3 it adds $0.8720 \to 0.8793$, i.e. $+0.007$. The reason is
visible in the importances — the string features cede their share as the CE
subsumes their signal. This was observed earlier too: a $+0.007$ improvement to
the CE gave the composition only $+0.001$.

The cross-encoder's train/val gap is $0.8292$ against $0.8003$, only $0.029$.
That is what determined how the superstructure was built: with such weak
memorization of the train set, the CE logit can be fed to the GBM directly.
Otherwise an out-of-fold scheme would be required — $K$ separate CEs, each
scoring its own held-out fold — since a GBM seeing unnaturally good scores on
train would overrate the feature. (This does not affect the honesty of the final
metric — neither the CE nor the GBM ever saw val — but the model would come out
suboptimal.)

### 6.5 Feature importances

Three snapshots showing how the contributions were redistributed:

| Feature | strings only | + CE v2 | + CE v3 (pretrained) |
|---|---:|---:|---:|
| `ce_score` | — | 0.2716 | **0.3539** |
| `number_jaccard` | — | — | 0.0023 |
| `category_code` | — | — | 0.0033 |
| `name_token_jaccard` | 0.1337 | 0.0270 | 0.0026 |
| `common_word_count` | 0.0965 | 0.0140 | 0.0006 |
| `attr_common_key_count` | 0.0382 | 0.0019 | 0.0006 |
| `name_seq_ratio` | 0.0316 | 0.0084 | 0.0017 |
| `attr_value_jaccard` | 0.0237 | 0.0099 | 0.0020 |
| `category_match`, `attrs1_empty`, `attrs2_empty` | 0.0000 | 0.0000 | 0.0000 |

`category_match` is identically zero in every configuration — candidates are
generated within a category, so the feature is constant. What is useful is not
the *consistency* of the categories but *which* category it is (`category_code`,
§2.3).

The important thing in this table is not the individual numbers but the trend:
as the cross-encoder gets stronger, the string features steadily cede their
share to it, and on top of CE v3 they all drop to noise level ($\leq 0.003$).
Collectively they still contribute $+0.007$, but the model now leans almost
entirely on a single feature.

---

## 7. Inference budget and the cross-encoder

### 7.1 Revisiting the initial premise

The pipeline was built around the thesis "speed is a hard constraint, so cheap
features only." The published limits refute that thesis:

| Evaluation environment resource | |
|---|---|
| CPU | 20 cores |
| RAM | 200 GB |
| GPU | **NVidia H100 80 GB** |
| Check / Public / Private limit | 1 / 6 / **13** min |
| Solution archive / docker image | 5 GB / 15 GB |

Measuring the current solution at a volume comparable to the test set (365K
pairs, 711K products): catalogue loading + features + GBM fit within
**~60–90 s** against a limit of **780 s**, with the GPU completely idle.

In other words, the optimization was targeting a resource that is abundant,
while the result (ROC-AUC 0.75 on string heuristics) was bottlenecked by feature
expressiveness. The order of magnitude of the headroom is roughly $10\times$ in
time, plus an unused H100.

**Measured** on the unpacked solution archive — that is, exactly what the
evaluation system will run. e5-base, 365,654 pairs, H100, with the card also
busy with a concurrent training run, so this is an **upper** estimate:

| Stage | Time | Share |
|---|---:|---:|
| Loading the catalogue and pairs | 4.7 s | 1% |
| Cross-encoder | 241.5 s | 72% |
| String features | 86.2 s | 26% |
| GBM and writing output | ~1 s | — |
| **Total** | **333 s** | against a 780 s limit |

Two practical conclusions follow.

**The string features have stopped paying for themselves.** They take 26% of the
time and add $+0.0018$ to the metric. While there is headroom, let them run —
but they are the first thing to drop if time becomes scarce.

**Model cost should be judged by computation, not parameter count.**
Multilingual models have a 250K-token vocabulary, and the embeddings consume most
of the parameters without requiring any computation:

| | Layers | Dimension | Total params | In layers (what counts) |
|---|---:|---:|---:|---:|
| MiniLM-L12 | 12 | 384 | 118M | ~21M |
| e5-base | 12 | 768 | 278M | ~85M |
| e5-large | 24 | 1024 | 560M | ~302M |

Between e5-base and e5-large the parameter difference is twofold, but the
computational difference ($\text{layers} \times \text{dimension}^2$) is
**3.5×**. Training time and inference time scale accordingly — by 3.5×, not 2×.

The model weights (490 MB for MiniLM, 1.1 GB for e5-base) fit comfortably within
the archive (5 GB) and image (15 GB) limits. The model must live **inside the
archive**: the evaluation environment has no network, so downloading from the HF
Hub at runtime will not work there.

### 7.2 Why a cross-encoder and not a bi-encoder

A bi-encoder encodes products independently and takes a cosine; a cross-encoder
feeds both texts into a single sequence via `[SEP]` and sees their interaction at
every attention layer.

The usual argument for a bi-encoder is reuse: a product's embedding is computed
once and participates in many pairs. That argument does not hold here. With 365K
pairs and 711K products, each product appears on average

$$\frac{2 \times 365\,654}{711\,304} \approx 1.03 \text{ times},$$

so there is essentially no reuse. A bi-encoder would require 711K forward passes,
a cross-encoder 365K (albeit on sequences twice as long), at noticeably higher
accuracy. The choice is unambiguous.

### 7.3 How it is integrated

The cross-encoder is an `AutoModelForSequenceClassification` with a single logit,
trained with binary cross-entropy on the **same** train split as the GBM
(`src/split.py` is the single source of truth for the split). Its logit is added
as one more feature column, with the same GBM on top.

The reason for composition rather than replacing the GBM with the CE is that the
features complement each other: exact matches on `артикул`/`oem-номер` are
precisely the signal a symbolic model catches more reliably than a transformer
with limited context and aggressive tokenization of alphanumeric codes.

An example from the data that explains why semantics are needed at all:

| Product A | Product B | String similarity |
|---|---|---|
| очки … romeo 1.56 hmc/emi **−2.00** рц 62-64 | **фотохромные** очки … romeo 1.56 **−3.00** рц 64-66 | very high |
| зубная щетка r.o.c.s. **black** edition classic | зубная щетка r.o.c.s. **red** edition classic | very high |
| сандалии fre gamo, цвет **чёрный** | fre gamo / сандалии, цвет **жёлтый** | very high |

All three pairs are almost certainly **not** duplicates, yet they differ by one
or two tokens inside a long shared text. Jaccard and Ratcliff–Obershelp both give
roughly one here; telling them apart is only possible by understanding that
dioptres, colour and edition are identifying attributes, not noise.

---

## 8. What's next

Done and closed: pretraining on the LLM labels (§6.2, the largest gain),
features from error analysis (§2.3), switching to the competition metric (§5.3),
calibration against the leaderboard (§6.3).

**What remains along the LLM-label axis.** The volume is exhausted — all 11.19M
pairs are in use. What remains are ways to squeeze more out of the same data:
- **a second pretraining epoch**: right now each pair is seen exactly once;
- **a larger model**: `multilingual-e5-base` (278M vs 118M). On an H100 it slows
  training by only 1.9× despite three times the parameters — MiniLM with
  dimension 384 underutilizes the card, while 768 uses it more efficiently;
- **selection by agreement**: keep only those LLM pairs where a model trained on
  human labels agrees with the LLM label;
- **calibration**: estimate $P(y = 1 \mid t)$ empirically on the intersection
  with the human labels, instead of assuming that $t$ is already a probability.

**Along the metric axis.** The metric averages PR-AUC over categories, and the
weakest categories are weak not because of poor ranking: Shoes has ROC-AUC
$0.872$ at PR-AUC $0.443$. What needs improving there is precision at the top of
the list rather than overall quality — that is, per-category threshold
calibration and features that give a confident "no" (differing SKUs under similar
names; right now `id_attr_match` only covers agreement).

**Along the feature axis:**
- drop `category_match` as a constant (identically zero by permutation
  importance in every configuration);
- TF-IDF cosine with catalogue-wide IDF weights: rare tokens (SKUs, models)
  should outweigh frequent ones (`для`, `шт`, `комплект`);
- numeric features from names with unit parsing — right now numbers are compared
  as strings, with no understanding that $0.5$ L and $500$ ml are the same.

**Along the reliability-of-conclusions axis.** Every decision was made on a
single split (`SPLIT_SEED = 1234`), and there have been more than twenty such
decisions. Large jumps cannot be explained by noise, but differences within
$\pm 0.005$ between close candidates do not deserve trust without a check on a
second seed.

---

## 9. Layout and how to run

```
matching_gbm/
├── run.py                 # inference, the competition CLI contract
├── metadata.json          # docker image + entry point
├── requirements.txt
└── src/
    ├── features.py        # per-pair features + encoder text (§2)
    ├── pipeline.py        # catalogue loading, chunking, union-find (§5.2)
    ├── split.py           # shared group-aware split for GBM and CE (§5.2)
    ├── train.py           # splitting, λ_H sweep, GBM training (§3–6)
    ├── ce_train.py        # cross-encoder fine-tuning (§7)
    └── ce_score.py        # scoring pairs with the cross-encoder (§7)
```

**Training the GBM.** Given §6, the baseline configuration uses human labels
only:

```bash
python3 -m src.train \
  --items_human_path items_human.parquet \
  --matches_path matches.parquet \
  --output_model_path gbm_model.joblib \
  --skip_llm
```

**Training the cross-encoder** (validated on the same split as the GBM):

```bash
python3 -m src.ce_train \
  --items_human_path items_human.parquet \
  --matches_path matches.parquet \
  --output_dir ce_model \
  --batch_size 64 --epochs 2
```

**GBM on top of the CE.** Score first, then train with `ce_score` as a feature:

```bash
python3 -m src.ce_score \
  --items_path items_human.parquet \
  --matches_path matches.parquet \
  --ce_model_dir ce_model \
  --output_path ce_scores.npy

python3 -m src.train \
  --items_human_path items_human.parquet \
  --matches_path matches.parquet \
  --ce_scores_path ce_scores.npy \
  --output_model_path gbm_model.joblib \
  --skip_llm
```

**Inference.** `run.py` picks up `ce_model/` if the directory exists, and runs on
string features alone if it does not:

```bash
python3 run.py \
  --items_path <test_items.parquet> \
  --matches_path <test_matches.parquet> \
  --output_path submission.csv
```

### Engineering notes

**Output alignment.** `build_feature_matrix` returns exactly `len(match_df)` rows
in the original order. Pairs whose product is missing from the catalogue get a
row of NaNs but are **not dropped** — otherwise the predictions shift relative to
`id1`/`id2`, and the submission ends up with fewer rows than the input file.

**Memory.** The catalogue is stored as a dict `id → (name, attributes, category)`
holding raw strings; JSON parsing and tokenization are done lazily, at the level
of a 200K-pair chunk, with a cache that lives only within the chunk. An early
version prepared all 12.4M products up front and hit the container limit (85 GB)
— at 13M products, heavyweight Python objects (`frozenset`, `dict`) impose a
multiple-fold overhead on top of the raw data.

**Why single-threaded.** Parallelizing via `multiprocessing.Pool` with `fork`
looked free (copy-on-write should have shared the catalogue between processes),
but in CPython any access to an object increments its reference count — that is,
it is a **write**. Pages of the "shared" dict are copied on read, reproducing
exactly the memory blow-up that fork was supposed to avoid; and with a large
parent RSS, `fork` itself further degrades on copying page tables. Genuinely
parallelizing this step requires the data in real shared memory (Arrow IPC /
`shared_memory`), not ordinary Python objects.

**A single split.** The GBM and the cross-encoder must be validated on
byte-for-byte identical held-out pairs, otherwise their metrics are incomparable
and, when their outputs are combined, a pair held out by one model but trained on
by the other leaks. That is why the split lives in `src/split.py` as the single
source of truth instead of being reproduced in every script.

**Consistency of the CE scores.** `--ce_scores_path` takes a `.npy` aligned
row-by-row with the label file **as it sits on disk**. The length is checked
explicitly: otherwise a stale score file, computed on a different or
already-filtered sample, would shift every value by an unknown offset — silently,
with no error. For the same reason, filtering by `exclude_items` is applied
**after** the features are built.

**Lazy torch import.** `run.py` imports `src.ce_score` only when `ce_model/` is
present. With the Check stage limited to one minute, the fixed overheads
(`import torch`, CUDA initialization, loading the weights) are a noticeable share
of the budget, and a solution without the cross-encoder should not have to pay
them.

**Compatibility check.** Before predicting, the model's `n_features_in_` is
verified against the width of the feature matrix: a mismatch between the GBM and
`ce_model/` (for example, a GBM trained with `ce_score` while the weights
directory did not make it into the archive) fails with a clear message instead of
silently producing wrong predictions.
