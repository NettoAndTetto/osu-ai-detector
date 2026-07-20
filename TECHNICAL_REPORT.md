# Auditable Detection of Mapperatorinator-Generated osu! Beatmaps

## A Multi-Channel System Based on Provenance, Source-Model Agreement, Generator Traces, and Mapping Structure

**Technical report, version 1.0 - 20 July 2026**  
**Author:** NettoAndTetto  
**Project:** `osu-ai-detector`

---

## Abstract

We present a conservative and auditable system for detecting residual evidence of Mapperatorinator generation in osu!standard beatmaps. The deployed system intentionally avoids a single overall verdict and instead reports four independent views: source-fact verification, agreement with five released Mapperatorinator source models, generator-trace forensics, and mapping-structure analysis. The two map-only statistical channels use extremely randomized tree ensembles over non-overlapping feature families, while the source-model channel teacher-forces the submitted beatmap under V29, V30, V31, V32, and V32-mini and combines token-level likelihood, rank, entropy, decoding-policy, and conditional-discrepancy measurements with a regularized logistic ranking head. All statistical scores are calibrated against 3,000 human beatmaps from unique BeatmapSetIDs using smoothed upper-tail ranks and Neyman-Pearson order-statistic thresholds; they are not interpreted as probabilities of AI authorship. In a one-shot final evaluation on a separate 3,000-map human set, the observed high-threshold false-positive rates were 0/3,000 for mapping structure, 2/3,000 for generator traces, and 0/3,000 for source-model agreement; the corresponding elevated-or-higher rates were 0.600%, 0.500%, and 0.567%. On ten held-out source-song clusters, generator traces reached the high threshold on all ten clusters, mapping structure on four, and source-model agreement on none, demonstrating that a more computationally expensive white-box method is not automatically a more sensitive detector at a fixed low-false-positive operating point. The system is therefore positioned as a versioned technical screening instrument for human review, not as a universal detector or an automated enforcement mechanism.

**Keywords:** AI-generated beatmaps, source-model detection, machine-generated sequence detection, conformal calibration, Neyman-Pearson classification, digital forensics, osu!, Mapperatorinator

---

## 1. Introduction

Generative models can now produce structured creative artifacts rather than only natural-language text or images. osu! beatmaps are one such artifact: a beatmap is a timed sequence of circles, sliders, spinners, inherited timing points, positions, and other events aligned to music. Mapperatorinator is an open-source generative system that maps audio context to tokenized beatmap events. Its outputs can be edited, reserialized, partially spliced into human maps, or stripped of explicit attribution before distribution.

Detecting such outputs is not the same problem as determining an author's intent. A final `.osu` file contains neither a complete editing history nor a reliable statement of how it was produced. The strongest defensible claim is therefore narrower: whether the file contains evidence consistent with specified, reproducible Mapperatorinator revisions and checkpoints.

The contributions of the project are:

- a four-channel detection formulation specialized to Mapperatorinator-generated osu!standard beatmaps;
- a source-grouped, version-pinned training corpus spanning five production generator variants, deterministic editing stress tests, and never-fit LoRA robustness cohorts;
- an audio-conditioned white-box detector adapted from likelihood-, rank-, conditional-curvature-, and temperature-normalization ideas in machine-generated text detection;
- independent 3,000-map calibration and one-shot 3,000-map final human evaluation splits;
- fail-closed model, tokenizer, search-protocol, and revision identity checks; and
- a local streaming implementation that avoids repeated GPU model switching and avoids returning one monolithic browser response.

The central empirical lesson is deliberately modest. Several Mapperatorinator families leave strong detectable traces, but no channel is uniformly best. Mechanical forensics is highly sensitive to uncleaned revisions and held-out LoRA outputs, while source-model agreement is computationally richer but conservative at the frozen low-FPR thresholds. Deep editing, short splices, future checkpoints, and distribution shift remain open problems.

## 2. Scope, Terminology, and Threat Model

### 2.1 Supported object

The calibrated target is an osu!standard (`mode 0`) `.osu` beatmap. The parser supports common v14 sections, red and inherited timing points, circles, sliders, spinners, and holds while preserving raw floating-point strings and slider control points. Statistical analysis requires enough events to form stable windows; very short or sparse maps can therefore abstain.

### 2.2 Positive concept

The system tests for evidence associated with specified Mapperatorinator V29-V32 families, including V29, V30, V31, V32 full, and V32-mini source-model checkpoints. It does **not** claim to detect every AI beatmap generator, every future Mapperatorinator revision, or AI assistance in general.

Let $x$ denote the final beatmap file and $a$ the audio referenced by that beatmap. The four deployed channels return separate outputs

$$
R(x,a) = \bigl(R_{\mathrm{src}}(x), R_{\mathrm{wb}}(x,a), R_{\mathrm{for}}(x), R_{\mathrm{con}}(x)\bigr),
$$

where `src`, `wb`, `for`, and `con` denote source facts, white-box source-model agreement, generator-trace forensics, and mapping content respectively. There is no deployed function $g(R)$ that converts this tuple into a total judgment.

### 2.3 Decision vocabulary

The statistical channels use three operational states:

- **high anomaly:** the frozen score strictly exceeds the supported high Neyman-Pearson threshold;
- **near threshold / elevated:** the score exceeds the supported elevated threshold but not the high threshold;
- **insufficient evidence:** neither threshold is exceeded.

A fourth state, **unable to determine**, is used when calibration is not applicable or the required protocol is incomplete. This is an abstention, not evidence of human authorship.

The source-fact channel uses categorical facts rather than a statistical scale: explicit disclosure, exact registered revision, no registered source fact, or unavailable identity information.

### 2.4 Threat model

The intended setting is post hoc inspection of a final beatmap. The system assumes that an analyst can obtain the `.osu` bytes and, for white-box analysis, the exact audio referenced by `AudioFilename`. It considers the following transformations:

- removal of creator, tag, editor, or template metadata;
- editor reserialization and timing-point normalization;
- coordinate perturbation or global reflection;
- removal of a generator-specific floating-point residue;
- partial insertion of generated segments into a human map; and
- official style-LoRA adapters applied to supported base checkpoints.

The system does not claim robustness to an adaptive attacker who has the detector, its thresholds, and unlimited freedom to remap the output. It also does not infer intent, misconduct, or policy violations.

## 3. Related Work

### 3.1 Source generator and structured sequence setting

Mapperatorinator is an open-source, audio-conditioned framework for generating tokenized osu! beatmap events [1]. Unlike natural language, its output is constrained by an event grammar, timing contexts, coordinate quantization, decoding processors, and editor serialization. These constraints create both opportunities and hazards for detection: source-model statistics are available, but naive reuse of natural-language thresholds is invalid.

### 3.2 Classifier two-sample detection

The mapping-structure and generator-trace channels can be viewed as classifier two-sample tests: a classifier learns a statistic that separates human and generated distributions, while held-out performance measures how much those distributions differ [2]. We use extremely randomized trees [3] because they model nonlinear interactions, support compact exported inference, and provide tree-path feature contributions. However, the classifier score is treated only as a ranking statistic; calibration and error control are performed on a separate human split.

### 3.3 Source-model detection

Machine-generated text detectors commonly use source-model likelihood, rank, entropy, or local probability geometry. DetectGPT uses curvature of the source model's log-probability surface [4]. Fast-DetectGPT replaces perturbation-heavy curvature estimation with a conditional probability discrepancy [5]. DetectLLM-LRR combines negative log likelihood and token rank [6]. TempTest targets normalization distortions induced by temperature and truncated sampling [7].

Our white-box channel adapts these ideas to Mapperatorinator's finite event vocabulary and grammar. The adaptation is not a direct application of published natural-language thresholds. All features are passed to a supervised ranking head, and the final statistic is independently calibrated on human beatmaps processed with the same audio-window search and checkpoint protocol.

### 3.4 Distribution-free calibration and asymmetric error control

Smoothed upper-tail ranks follow the general split-conformal principle of calibrating a fixed score on exchangeable held-out data [8]. The Neyman-Pearson umbrella approach selects a class-0 order statistic so that the type-I error exceeds a target rate with probability at most $delta$ [9]. This asymmetric objective matches the project's intended use: a false accusation is considered substantially more costly than a missed detection.

## 4. System Overview

The deployed analysis order is:

1. **Source-Fact Verification** - explicit AI disclosure and exact known-revision identity.
2. **Mapperatorinator Source-Model Agreement** - token-level agreement with five released generator checkpoints.
3. **Generator-Trace Forensics** - mechanical coordinate, timing, SV, and serialization traces.
4. **Mapping-Structure Analysis** - rhythm, movement, density, object, slider, and event-order structure.

The order is a presentation choice, not an evidential hierarchy. In particular, source-model agreement is computationally sophisticated but empirically less sensitive than trace forensics on several held-out cohorts.

### 4.1 Shared parsing and localization

Each file is parsed once. The map-only channels operate on 24-second windows with an 8-second stride and at least 12 objects per window. The white-box channel does not scan arbitrary audio and then select the largest score. Instead, a frozen label-free proposal rule uses the beginning, end, densest region, and the highest content-scoring non-overlapping regions. Candidate parameters are fixed to three content windows, 45 seconds for the dense region, 30-second edge regions, and a four-second merge gap. One audio context is selected per merged interval.

This restriction prevents a multiple-comparisons mismatch in which calibration maps are searched over a few windows but test maps are searched over the entire song.

### 4.2 Independent outputs

Every statistical result contains:

- the raw discriminative score;
- the smoothed human-null upper-tail $p$-value;
- the corresponding anomaly percentile;
- elevated and high thresholds and whether each was crossed;
- calibration size and model identity;
- the strongest local evidence; and
- explicit abstention and limitation fields.

The anomaly percentile is a reference rank, not a probability of AI authorship. A map may be more anomalous than 99% of the calibration set and still remain below a finite-sample NP threshold selected to control a 1% target FPR with high probability.

## 5. Detection Methods

### 5.1 Channel 1: Source-Fact Verification

The source-fact channel performs two direct checks.

#### Explicit disclosure

Creator, Version, and Tags fields are searched for explicit Mapperatorinator, osuT5, or AI-generated declarations. Such a match is a fact about the submitted file's text, not a learned prediction.

#### Exact revision registry

Let $h(x)=\mathrm{SHA256}(x)$, and let $mathcal{R}_{+}$ be the frozen set of verified positive file revisions. The registry indicator is

$$
E_{\mathrm{registry}}(x) = \mathbf{1}\{h(x)\in\mathcal{R}_{+}\}.
$$

The identity key is the complete file hash. A file with the same BeatmapID or BeatmapSetID but a different SHA-256 does not inherit the positive label. This prevents an edited replacement from being treated as the historically reported revision.

Source facts are high-precision but fragile. Metadata can be removed, and any byte-level edit changes the registry hash. For that reason this channel is displayed independently from statistical evidence.

### 5.2 Channel 2: Mapperatorinator Source-Model Agreement

This channel asks whether the observed beatmap events look unusually probable under the released Mapperatorinator generators when conditioned on their actual audio and preceding event context.

#### Checkpoint set and execution contract

The frozen protocol requires exactly five checkpoints: V29, V30, V31, V32 full, and V32-mini. It binds model weights, tokenizer bytes, vendor runtime source, candidate intervals, precision (`bf16`), attention implementation (`sdpa`), forward batch size (`1`), and the ordered interval-to-result mapping. Missing, additional, or changed identities preserve descriptive output but invalidate the calibrated decision.

#### Per-token quantities

For an observed token $y_i$, context $c_i$, logits $z_i(v)$, and probability

$$
p_i(v)=\frac{\exp z_i(v)}{\sum_u \exp z_i(u)},
$$

the system records:

$$
\mathrm{NLL}_i=-\log p_i(y_i), \qquad
r_i=1+\sum_v \mathbf{1}\{z_i(v)>z_i(y_i)\},
$$

as well as entropy, target margin, nucleus membership, retained mass, and EventType-family-conditioned counterparts. Full-vocabulary statistics are primary; family-conditioned quantities that use the observed target family after the fact are explicitly marked as heuristic.

The DetectLLM-style log-likelihood/log-rank ratio is

$$
\mathrm{LRR}=\frac{\frac{1}{m}\sum_{i=1}^{m}\mathrm{NLL}_i}
{\frac{1}{m}\sum_{i=1}^{m}\log r_i}.
$$

When all observed tokens have rank one, the denominator is zero and the feature is marked undefined rather than stabilized with an arbitrary epsilon.

For the Fast-DetectGPT-style conditional discrepancy, define

$$
\mu_i = \mathbb{E}_{V\sim p_i}[\log p_i(V)], \qquad
\sigma_i^2 = \mathrm{Var}_{V\sim p_i}[\log p_i(V)].
$$

The sequence statistic is accumulated before standardization:

$$
D_{\mathrm{FD}}=
\frac{\sum_{i=1}^{m}\left(\log p_i(y_i)-\mu_i\right)}
{\sqrt{\sum_{i=1}^{m}\sigma_i^2}}.
$$

This is not the mean of independently standardized token $z$-scores; token-level quantiles are retained only as auxiliary features.

For a tested sampling temperature $	au$, TempTest uses

$$
\log Z_{i,\tau}=\log\sum_v p_i(v)^{1/\tau},
$$

$$
T_{i,\tau}=\log Z_{i,\tau} - \left(\frac{1}{\tau}-1\right)\log p_i(y_i).
$$

The detector uses $	au=0.9$ for the main event stream and $	au=0.1$ for the relevant timing context, matching the frozen generation configuration. Grammar masks, monotonic constraints, lookback bias, top-$k$, and top-$p$ policy effects are replayed and separately audited.

#### Supervised aggregation

Each checkpoint-window observation is flattened to a 604-feature vector with explicit missingness indicators. After frozen imputation and scaling, an L2-regularized logistic head produces

$$
s_{cw}=\sigma\!\left(b+\boldsymbol{\beta}^{\top}\mathbf{z}_{cw}\right).
$$

The map score is the mean of the two highest successful checkpoint-window scores. The sigmoid is a ranking score, not an authorship posterior. The selected regularization parameter is $C=0.03$, chosen by five-fold source-BeatmapSetID grouped cross-validation.

### 5.3 Channel 3: Generator-Trace Forensics

The forensic channel uses only an allow-listed set of mechanical features associated with known generator and postprocessor behavior. It excludes semantic rhythm and movement features so that its meaning remains narrow: similarity to an insufficiently cleaned Mapperatorinator revision.

Feature families include:

- coordinate residues and decoder grids, including the V29 32-pixel cell-centre residue;
- millisecond timing residues and time lattices;
- inherited timing-point and slider-velocity floating-point serialization;
- editor templates and output ordering; and
- registered postprocessor patterns.

For example, V29 combined-position tokens use 32-pixel cells and, without refinement, decode to cell centres. The detector measures

$$
q_{32,16}=\frac{1}{N}\sum_{j=1}^{N}\mathbf{1}\{c_j\bmod 32=16\}
$$

for all coordinates and separately for object heads. The exact signature requires sufficient sample count and simultaneous concentration in both views. V30/V31 four-pixel grids and V32 even-coordinate patterns are treated more cautiously because ordinary editor snapping can reproduce them.

Another source-level residue comes from writing inherited beat length as approximately $-100/q+10^{-10}$ for quantized slider velocity $q$. The detector reverses the transformation and counts values whose residual lies in the frozen interval. Such residues are useful but removable by reserialization.

Each 24-second window is scored by a 220-tree ExtraTrees ensemble with maximum depth 16, minimum leaf size 3, and feature fraction 0.7. The map score is the mean of the top three window scores.

### 5.4 Channel 4: Mapping-Structure Analysis

The content channel measures whether the beatmap's structure resembles the generated development distribution relative to human maps. It intentionally excludes Creator, Tags, Editor templates, and raw floating-point serialization.

Its representation contains:

- rhythm and inter-object interval statistics;
- movement distance, angle, and direction change;
- object, slider, timing, and local-density structure; and
- a 512-dimensional hashed canonical event $n$-gram representation.

Each window is scored by a 180-tree ExtraTrees ensemble with maximum depth 18, minimum leaf size 2, feature fraction 0.7, and top-three-window aggregation. The ensemble window score is

$$
s_w=\frac{1}{T}\sum_{t=1}^{T} f_t(\phi(x,w)),
$$

and the map score is

$$
S(x)=\frac{1}{k}\sum_{j=1}^{k}s_{(j)}, \qquad k=\min(3,|W_x|),
$$

where $s_{(1)}\geq s_{(2)}\geq\cdots$ are sorted window scores.

To avoid treating any unfamiliar map as AI-generated, the system computes a robust out-of-distribution distance. For feature vector $mathbf{x}$, frozen human median $mathbf{m}$, and robust scales $mathbf{a}$,

$$
z_j=\frac{|x_j-m_j|}{a_j}, \qquad
d_{\mathrm{OOD}}=\sqrt{\frac{1}{d}\sum_{j=1}^{d}\min(z_j,20)^2}.
$$

If any selected window exceeds the frozen support boundary, the content decision abstains. The report also records the fraction of dimensions with $z_j>8$.

## 6. Training, Calibration, and Frozen Protocol

### 6.1 Human splits

The production human manifest contains mutually disjoint BeatmapSetIDs:

| Split | Unique sets / maps | Purpose |
|---|---:|---|
| Development | 1,000 preregistered; 990 directly fit | Feature, model, and aggregation selection; grouped OOF scores |
| Calibration | 3,000 | Human-null ranks, OOD support, and NP thresholds |
| Final test | 3,000 | One-shot false-positive and abstention evaluation |

Ten source-heldout set IDs are preregistered and excluded from the 1,000-map development split. Calibration and final test are rejected by training code. Final-test identities remain inaccessible until the complete evaluation freeze is created.

### 6.2 Generated training corpus

For each source set, the system uses the same fixed osu!standard reference difficulty and audio while varying generator revision, seed, and requested difficulty. The released corpus contains:

| Generator family | Planned | Usable | Development / held out |
|---|---:|---:|---:|
| V30, revision `2025-08-27` | 120 | 120 | 90 / 30 |
| V31, revision `2025-10-31` | 120 | 119 | 89 / 30 |
| V32 pre-halfstep | 120 | 119 | 89 / 30 |
| V32 current full | 120 | 119 | 89 / 30 |
| V32 current mini | 120 | 120 | 90 / 30 |
| **Total** | **600** | **597** | **447 / 150** |

The three failed planned conditions are preserved as fail-closed attrition records rather than silently replaced. Each usable base generation has ten deterministic training stress variants, yielding 5,970 stress rows. All seeds, versions, references, and transformed descendants from the same source set remain in the same group.

The held-out external evaluation contains 150 base generations over ten source-song clusters and 1,800 evaluation-only stress variants. A separate never-fit cohort contains 80 maps from four official V30/V32 style-LoRA configurations over the same ten source songs and two seeds. These external cohorts are excluded from feature selection, model selection, OOD support, and threshold calibration.

### 6.3 Model selection

Content and forensic ExtraTrees hyperparameters are selected with five outer source-group folds and four inner source-group folds. The final development out-of-fold results were:

| Channel | ROC-AUC | Standardized partial ROC-AUC at 1% FPR |
|---|---:|---:|
| Mapping structure | 0.987331 | 0.804675 |
| Generator traces | 0.997582 | 0.963086 |
| Source-model agreement | 0.947244 | 0.757083 |

For source-model agreement, TPR at 1% development FPR was 0.611693 and TPR at 0.1% development FPR was 0.318242. These are development OOF figures, not final blind-test estimates.

### 6.4 Smoothed human-null rank

For a fixed map score $S(x)$ and calibration scores $H_1,\ldots,H_n$, the upper-tail rank is

$$
p_{\mathrm{human}}(x)=
\frac{1+\sum_{i=1}^{n}\mathbf{1}\{H_i\geq S(x)\}}{n+1}.
$$

With $n=3{,}000$, the minimum attainable value is $1/3001\approx0.00033322$. The displayed anomaly percentile is $100(1-p_{\mathrm{human}})$. This percentile is a location within the frozen human reference distribution, not $P(\mathrm{AI}\mid x)$.

### 6.5 Neyman-Pearson thresholds

Let $H_{(1)}\leq\cdots\leq H_{(n)}$ be the sorted calibration scores. For target human false-positive rate $alpha$, choose $k$ such that

$$
\Pr\{\mathrm{FPR}(H_{(k)})>\alpha\}
\leq
\sum_{j=k}^{n}{n\choose j}(1-\alpha)^j\alpha^{n-j}
\leq\delta.
$$

The decision rule is strictly $S(x)>H_{(k)}$. Equality does not trigger, because changing `>` to `>=` would invalidate the order-statistic guarantee in the presence of ties.

The frozen operating points are:

| Channel | Elevated threshold (α=1%, δ=5%) | High threshold (α=0.1%, δ=5%) |
|---|---:|---:|
| Mapping structure | 0.8536082900 | 0.9821269935 |
| Generator traces | 0.6656215754 | 0.9119965306 |
| Source-model agreement | 0.9824260729 | 0.9999999992 |

For $n=3{,}000$, the elevated threshold is the 2,980th order statistic, with violation-probability bound 0.034615. The high threshold is the maximum calibration score, with bound 0.049712. This explains why a visually striking 99th-percentile result may still be below the supported elevated threshold.

### 6.6 One-shot evaluation integrity

The final evaluation uses an external-first sequence:

1. freeze source, model, calibration, checkpoint, tokenizer, and manifest identities;
2. evaluate all external generated and robustness cohorts;
3. write and seal an external receipt;
4. atomically consume a one-shot final authorization;
5. first parse the sealed 3,000-map final manifest;
6. evaluate and persist map-level outputs;
7. write the summary and completion seal; and
8. publish an exact-member package.

The four channels remain separate throughout evaluation. No post-freeze OR statistic is constructed, because an OR of separately calibrated tests would create a new statistic with different false-positive behavior.

## 7. Results

### 7.1 Final human false-positive and abstention rates

The final human set contains 3,000 beatmaps from 3,000 unique BeatmapSetIDs, 1,576 unique creators, and a maximum creator cluster of 33 maps. The table below reports the three independently calibrated statistical channels.

| Channel | High events / FPR | Elevated-or-higher events / FPR | Abstentions |
|---|---:|---:|---:|
| Mapping structure | 0/3,000; 0.000% (95% CP 0.000-0.123%) | 18/3,000; 0.600% (0.356-0.947%) | 3/3,000; 0.100% |
| Generator traces | 2/3,000; 0.067% (0.008-0.241%) | 15/3,000; 0.500% (0.280-0.823%) | 0/3,000; 0.000% |
| Source-model agreement | 0/3,000; 0.000% (0.000-0.123%) | 17/3,000; 0.567% (0.330-0.906%) | 0/3,000; 0.000% |

The one-sided 95% upper bound for a zero-event high FPR is approximately 0.100%. Confidence intervals are exact Clopper-Pearson intervals [10]. These observations do not establish a zero population FPR. They rely on the exchangeability of calibration and target human maps. Creator repetition, era, mapping style, and toolchain can violate that assumption; the creator-cluster audit describes this dependence but cannot eliminate it.

The three mapping-structure abstentions were outside the frozen content calibration support. Missing decisions were never replaced with zero scores.

### 7.2 Source-fact verification

The exact production revision registry matched all 8/8 available verified positive revisions in the frozen public-positive cohort. One known replacement revision did not inherit the historical label, and two reported revisions that could not be recovered were excluded rather than substituted. This is an identity result, not a generalization result: a single byte-level edit prevents an exact registry match.

### 7.3 Held-out base generations

Primary sensitivity is reported over ten independent source-song clusters; three seeds and multiple generator versions within a source song are dependent members.

| Channel | High clusters | Elevated-or-higher clusters | Abstaining clusters |
|---|---:|---:|---:|
| Mapping structure | 4/10 (40%) | 10/10 (100%) | 0/10 |
| Generator traces | 10/10 (100%) | 10/10 (100%) | 0/10 |
| Source-model agreement | 0/10 (0%) | 2/10 (20%) | 0/10 |

For ten clusters, the exact 95% Clopper-Pearson interval is wide: 100% corresponds to 69.150-100%, 40% to 12.155-73.762%, 20% to 2.521-55.610%, and 0% to 0-30.850%. The small number of independent songs is therefore a major uncertainty source.

The result also falsifies a tempting assumption: white-box access does not guarantee the strongest detector. At the frozen high threshold, source-model agreement detected none of the ten held-out base clusters, while mechanical forensics detected all ten. The white-box head still provides distinct token-level evidence and local explanations, but its current operating point is extremely conservative.

### 7.4 Never-fit style-LoRA robustness

Each adapter uses the same ten source songs, so rows across adapters are not 40 exchangeable song clusters. Results are shown per ten-set adapter cohort.

| Adapter cohort | Mapping structure high / elevated | Generator traces high / elevated | Source-model agreement high / elevated |
|---|---:|---:|---:|
| V30 `v30_2025` | 1/10 / 8/10 | 10/10 / 10/10 | 0/10 / 0/10 |
| V30 `v30_kroytz` | 1/10 / 10/10 | 10/10 / 10/10 | 0/10 / 0/10 |
| V32 `v32_kroytz` | 0/10 / 10/10 | 10/10 / 10/10 | 0/10 / 0/10 |
| V32 `v32_voxell` | 1/10 / 7/10 | 10/10 / 10/10 | 0/10 / 0/10 |

The external never-fit and training-leakage audits passed. These results indicate strong persistence of mechanical traces under the tested adapters and a substantial calibration gap for the white-box channel. They do not establish performance for arbitrary adapters or future checkpoint families.

## 8. Analysis and Discussion

### 8.1 Why a 99th-percentile result can remain below threshold

The interface displays both a rank percentile and a decision threshold. They answer different questions. The percentile indicates where a score falls among 3,000 human calibration maps. The threshold is an order statistic chosen to satisfy a finite-sample type-I-error guarantee. For the elevated target, the threshold is approximately the 99.33rd empirical percentile, not the 99th. For the high target, it is the calibration maximum. Therefore, “more anomalous than 99% of reference human maps” is compatible with “insufficient evidence under the frozen threshold.” This is intentional conservatism, not a color-mapping error.

### 8.2 Complexity is not evidential superiority

Source-model agreement has privileged access to the generator and provides the richest explanations, but it also faces the hardest calibration problem. Checkpoint mismatch, grammar constraints, top-$p$ and temperature policies, human maps that are themselves highly probable under the generator, and an extreme low-FPR threshold can all compress useful separation. The final data show that the white-box channel is not a replacement for simpler mechanical or structural evidence.

### 8.3 Why no total verdict is deployed

A total verdict hides several important distinctions:

- an exact known revision is identity evidence, while a statistical score is distributional evidence;
- mechanical traces can be strong but removable;
- structural similarity may survive reserialization but is more vulnerable to domain shift;
- source-model agreement requires correct audio, GPU computation, and exact protocol identity; and
- combining channels changes the null distribution and therefore requires separate calibration.

The current interface leaves these results independent. This allows users to see disagreement rather than forcing it into one probability-like number.

### 8.4 Interpretation of disagreement

Common patterns include:

- **source fact positive, statistical channels weak:** likely a known or self-declared file that was heavily edited;
- **trace positive, content and white-box weak:** likely preserved mechanical residues without strong semantic or likelihood agreement;
- **content elevated, trace weak:** structural similarity without known serialization residue, which may indicate editing, a different revision, or a human false positive;
- **white-box elevated, other channels weak:** unusual agreement with the source model that requires close inspection of windows, token families, and protocol compatibility; and
- **all channels weak:** insufficient evidence only, not proof of human authorship.

## 9. Engineering Design

### 9.1 GPU-friendly checkpoint-major scheduling

A naive batch implementation could process one map at a time and repeatedly load V29, V30, V31, V32, and V32-mini. The deployed scheduler instead performs all CPU parsing and candidate selection first, then loads each checkpoint once and processes every pending map before moving to the next checkpoint. This changes model-loading complexity from approximately $O(MC)$ loads to $O(C)$, where $M$ is the number of maps and $C=5$ is the checkpoint count.

Only one GPU job runs across tasks, preventing competing allocations. Within a job, progress increments after each map-checkpoint unit, so the reported fraction reflects actual completed inference rather than file count alone.

### 9.2 Streaming API and bounded browser state

The local service uses asynchronous jobs:

- `POST /api/jobs` creates a job;
- `GET /api/jobs/{id}/events` streams progress and lightweight per-method cards via server-sent events;
- paginated result endpoints return completed lightweight results;
- full per-map JSON is downloaded on demand; and
- batch HTML and JSON reports are streamed as files rather than inserted into the page DOM.

Map-only results can appear before source-model analysis finishes. The white-box card is published only after all required checkpoints complete, avoiding a misleading interim verdict.

### 9.3 Local deployment and privacy

The interface is a self-contained local page with no external CDN, account system, database, or browser storage. Map-only channels run from pure JSON artifacts without scikit-learn, PyTorch, a GPU, or network access. Source-model agreement recommends CUDA; CPU execution is permitted with an explicit performance warning. The default server binds to loopback.

### 9.4 Validation

Validation covers parser and feature contracts, calibration behavior, source-model protocol identity, resumable inference, package integrity, and release installation. End-to-end browser tests verify streamed results, four independent methods, absence of an overall verdict, anomaly-percentile explanations, report downloads, accessible controls, and desktop and mobile layouts.

## 10. Limitations

1. **Generator scope.** The detector is specialized to known Mapperatorinator revisions and checkpoints. Results do not transfer automatically to other generators or future releases.
2. **Mode scope.** Only osu!standard is calibrated.
3. **Finite human reference.** A 3,000-map final set supports useful low-FPR estimates but cannot prove zero error under broad community deployment.
4. **Exchangeability.** Calibration guarantees assume calibration and target human scores are exchangeable. Repeated creators and unmeasured era or style differences may violate this assumption.
5. **Few independent generated sources.** Held-out sensitivity is based on ten source songs. Seeds, revisions, and adapters within a song do not create new independent songs.
6. **White-box sensitivity.** The current source-model agreement channel has low held-out and LoRA sensitivity at frozen thresholds despite rich token-level features.
7. **Editing and adaptive evasion.** Reserialization, remapping, coordinate perturbation, and short edited splices can substantially reduce detection.
8. **Exact identity fragility.** SHA-based registry evidence disappears after any byte-level change.
9. **Audio dependency.** White-box analysis requires the correct referenced audio. The local UI never guesses a substitute file when `AudioFilename` is missing; the user must select it.
10. **No posterior probability.** The data do not identify real-world AI prevalence, so neither sigmoid scores nor human-null percentiles are probabilities of AI authorship.
11. **Public-positive labels.** Public cases are identity-blind during learned-channel scoring but were visible during method research; they are not a statistically blind benchmark.

## 11. Ethics and Intended Use

This system is intended for technical screening and manual review. It should not be used as the sole basis for punishment, public accusation, ranking action, or claims about an author's intent. A high result means that a frozen statistic crossed a low-FPR threshold under stated assumptions; it does not establish provenance beyond reasonable doubt.

Recommended review practice is:

1. preserve the exact submitted bytes and hashes;
2. inspect revision history and mapset context;
3. distinguish direct source facts from statistical evidence;
4. inspect the localized windows and strongest feature or token explanations;
5. check whether the model and search protocol are exactly compatible;
6. consider known failure modes and abstentions; and
7. seek author clarification before any consequential action.

The underlying `.osu` files and audio may be copyrighted user content. Checksums and local manifests provide reproducibility without granting redistribution rights.

## 12. Reproducibility and Public Artifacts

The public release separates executable application code, research code, original model artifacts, and reproducibility metadata:

| Artifact | Public location | Contents |
|---|---|---|
| Application | [github.com/NettoAndTetto/osu-ai-detector](https://github.com/NettoAndTetto/osu-ai-detector) | Local inference service, Web interface, installer, and release tests |
| Research code | [github.com/NettoAndTetto/osu-ai-detector-research](https://github.com/NettoAndTetto/osu-ai-detector-research) | Training, calibration, evaluation, and acquisition entry points |
| Model artifacts | [huggingface.co/NettoAndTetto/osu-ai-detector-models](https://huggingface.co/NettoAndTetto/osu-ai-detector-models) | Calibrated detector parameters and exact artifact manifest |
| Reproducibility metadata | [huggingface.co/datasets/NettoAndTetto/osu-ai-detector-reproducibility](https://huggingface.co/datasets/NettoAndTetto/osu-ai-detector-reproducibility) | Registered split identities, source revisions, checksums, acquisition metadata, and aggregate reference results |

The repositories are versioned at `v1.0.0`. Third-party beatmaps, audio, and upstream Mapperatorinator weights are not redistributed. Acquisition scripts recover permitted source material from its original location and verify registered identities before use. The research repository provides the executable sequence in `REPRODUCTION_COMMANDS.md`; the reproducibility dataset records the immutable evaluation lineage and aggregate reference outputs.

The final model lineage hashes are:

| Role | Model ID | SHA-256 |
|---|---|---|
| Mapping structure | `mapperatorinator-content-v2` | `65976506cc4b3ad78913a6fff9cd25e7411c5b1e20dd6b3db2f3bea3d5ebf4ac` |
| Generator traces | `mapperatorinator-forensic-v3` | `e1bad048e90f5df00d05c4f1a26731da2c11966f5daa3ca823b40423881c852f` |
| Source-model agreement | `mapperatorinator-whitebox-logistic-v1-501c17853200` | `41c90501b1e37b23db53211c65ddaa4769cdfb71c633369c274cd800c954b347` |
| Exact revision registry | `public-incident-exact-revisions-2026-07-13` | `7a0179ec2c80a7ed9bc3796ca1e1a97583355c2b6e6f734300dbd28ac01b20cc` |

## 13. Conclusion

Mapperatorinator detection is best treated as a structured forensic problem rather than a binary authorship classifier. Direct provenance, source-model likelihood, mechanical generator traces, and semantic mapping structure provide complementary evidence and exhibit materially different failure modes. Independent human calibration allows conservative low-FPR thresholds, but the guarantees remain conditional on exchangeability and the frozen protocol.

The final evaluation supports three limited conclusions. First, the content and white-box channels produced no high-threshold events in 3,000 final human maps, while the forensic channel produced two. Second, mechanical traces were the strongest signal on the ten held-out source-song clusters and four tested LoRA adapters. Third, the source-model channel, despite its technical depth, was not a high-sensitivity detector at the current thresholds. These findings motivate the deployed design: four separate, interpretable results, explicit abstention, and no total verdict.

Future work should prioritize more independent source songs, broader human creator and era coverage, more checkpoint and adapter families, independently maintained edited positives, and calibration-preserving improvements to source-model statistics. Lowering thresholds to recover a small set of known cases would sacrifice the project's main safety objective and is not recommended.

---

## References

[1] OliBomby. **Mapperatorinator**. Open-source software repository. https://github.com/OliBomby/Mapperatorinator (accessed 20 July 2026).

[2] David Lopez-Paz and Maxime Oquab. **Revisiting Classifier Two-Sample Tests**. International Conference on Learning Representations, 2017. https://arxiv.org/abs/1610.06545

[3] Pierre Geurts, Damien Ernst, and Louis Wehenkel. **Extremely Randomized Trees**. *Machine Learning*, 63(1):3-42, 2006. https://doi.org/10.1007/s10994-006-6226-1

[4] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. **DetectGPT: Zero-Shot Machine-Generated Text Detection Using Probability Curvature**. ICML, PMLR 202:24950-24962, 2023. https://proceedings.mlr.press/v202/mitchell23a.html

[5] Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. **Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature**. ICLR, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html

[6] Jinyan Su, Terry Zhuo, Di Wang, and Preslav Nakov. **DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text**. Findings of EMNLP, pages 12395-12412, 2023. https://doi.org/10.18653/v1/2023.findings-emnlp.827

[7] Tom Kempton, Stuart Burrell, and Connor J. Cheverall. **TempTest: Local Normalization Distortion and the Detection of Machine-Generated Text**. AISTATS, PMLR 258:1972-1980, 2025. https://proceedings.mlr.press/v258/kempton25a.html

[8] Anastasios N. Angelopoulos and Stephen Bates. **A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification**. arXiv:2107.07511, 2021. https://arxiv.org/abs/2107.07511

[9] Xin Tong, Yang Feng, and Jingyi Jessica Li. **Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics**. *Science Advances*, 4(2):eaao1659, 2018. https://doi.org/10.1126/sciadv.aao1659

[10] Charles J. Clopper and Egon S. Pearson. **The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial**. *Biometrika*, 26(4):404-413, 1934. https://doi.org/10.1093/biomet/26.4.404
