# Room occupancy prediction: what this repo does, and a review

*Review of `occupancy_analysis_data.ipynb` as of commit `78fc1d0`. The notebook was not modified. The proposed additions are in [proposed_cells.ipynb](proposed_cells.ipynb).*

---

## 1. In one paragraph

The project predicts **how many people (0 to 3) are in a room** from 16 environmental sensors. The prediction drives a **ventilation decision**: run the ventilation at level 0, 1, 2 or 3, where too little air is expensive (people breathe bad air) and too much air wastes energy. Several decision trees are compared on a time-ordered train/validation/test split, judged by a cost-based metric (regret). The winner is a 3-sensor tree (candidate C). It is near-perfect on validation but misses a 95-minute episode on the test day, when 3 people were in the room with the lights almost off.

---

## 2. What is in the repo

| File | What it is |
|---|---|
| [occupancy_analysis_data.ipynb](occupancy_analysis_data.ipynb) | The whole analysis: 71 cells, 10 sections, all outputs saved. |
| [saved_choices.json](saved_choices.json) | Choices "frozen" before the test (written by the notebook in section 9): the final candidate, its sensors and hyperparameters, and those of the other candidates. |
| [requirements.txt](requirements.txt) | numpy, pandas, matplotlib, scikit-learn (pinned). |
| `.gitignore` | Ignores `.venv`, the downloaded CSV, `.DS_Store`. |
| [README.md](README.md) | Only a title ("Augmented_decision"). |
| **New:** [proposed_cells.ipynb](proposed_cells.ipynb) | Companion notebook with the proposed cells (section 7 of this report). |
| **New:** `review_report.md` | This file. |

The dataset is not committed. The notebook downloads it from UCI the first time it runs (`Occupancy_Estimation.csv`, git-ignored).

---

## 3. The dataset

**Source:** *Room Occupancy Estimation*, UCI Machine Learning Repository (dataset #864).

| | |
|---|---|
| Rows | 10,129 measurements, about one every 30 seconds |
| Period | 22–26 Dec 2017 and 10–11 Jan 2018 (7 dates, with a two-week gap) |
| Quality | No missing values, no duplicate rows, already in time order |
| Target | `Room_Occupancy_Count`: 0, 1, 2 or 3 people (recorded by hand) |

**The 16 sensors (the inputs):**

| Group | Columns | Unit |
|---|---|---|
| Temperature | `S1_Temp` … `S4_Temp` | °C |
| Light | `S1_Light` … `S4_Light` | lux |
| Sound | `S1_Sound` … `S4_Sound` | volts |
| CO₂ | `S5_CO2`, `S5_CO2_Slope` (how fast CO₂ changes) | ppm |
| Motion (PIR) | `S6_PIR`, `S7_PIR` | 0/1 |

**Three properties of this data drive every result:**

1. **Strong class imbalance.** The room is empty 81% of the time (0: 8,228 rows, 1: 459, 2: 748, 3: 694), so "always predict 0" is already 81% accurate. This is why the project uses regret and macro-F1, not accuracy.
2. **People are present on only 3 dates** (22 Dec, 23 Dec, 10 Jan). The other dates are nights or holidays with an empty room.
3. **The light sensors almost give away the answer.** In the December data, light is a very strong signal of occupancy. The final tree splits on two light sensors and one temperature sensor only. In the LASSO fit, the 4 light coefficients are at least 8 times bigger than those of any other sensor. Every model ends up relying on light, and that is what fails on the test day (section 7, 11.5).

---

## 4. The end goal: a decision, not just a prediction

The business framing is a company that wants to ventilate offices only as much as needed.

- **Action** `a` = ventilation level 0–3 (level `a` gives enough air for `a` people).
- **Cost** of choosing `a` when there are really `y` people:

  `L(a, y) = 1 × a  +  2 × max(y − a, 0)`

  Each level of energy costs 1, and each person without enough air costs 2. The team chose these two numbers.
- **Decision rule.** A model outputs probabilities for 0/1/2/3 people, and the action with the **lowest expected cost** is chosen. With costs 1 and 2, this means: *ventilate for at least k people as soon as P(k or more people) > 1/2*. In other words, the action is the **median** of the predicted distribution.
- **Metric: regret** = mean cost − cost with perfect information. 0 is perfect. Regret is used because the three periods contain different numbers of people, so raw costs are not comparable. The perfect-information costs are 0.699 (train), 0.481 (validation) and 0.164 (test).

---

## 5. What the notebook does, step by step

| § | Step | Main result |
|---|---|---|
| 1–3 | Load, check, explore | Clean data; 81% empty; light and temperature sensors strongly correlated (mean \|r\| 0.69 and 0.85) |
| 4 | **Time-ordered split** (not random: rows 30 s apart are near-duplicates) | train: 22 Dec 10:49 → 23 Dec 15:00 (3,200 rows) · validation: → 25 Dec (2,105) · test: 25 Dec → 11 Jan (4,824, **no row with exactly 1 person**) |
| 5–6 | Cost matrix, decision rule, metrics | As in section 4 above |
| 7 | Baselines | "Never ventilate" (validation regret 0.481) and "always ventilate for 3" (2.519) |
| 8 | **Four candidates**, all decision trees, each tuned on a grid of `max_depth` × `min_samples_leaf` by validation regret | A: 16 raw sensors · B: 6 aggregated sensors · C: forward selection · D: LASSO selection |
| 9 | Pick the final model on validation only, freeze it in `saved_choices.json` | **C** (3 sensors: `S1_Light`, `S1_Temp`, `S2_Light`) |
| 10 | Test, once | C is still the best of the four, but it misses 184 rows with 3 people on 10 Jan |

**Results** (the last two rows are new, from [proposed_cells.ipynb](proposed_cells.ipynb)):

| Candidate | Sensors | Validation regret | Test regret |
|---|---:|---:|---:|
| Never ventilate | 0 | 0.481 | 0.164 |
| Always ventilate for 3 | 0 | 2.519 | 2.836 |
| A: tree, 16 raw sensors | 16 | 0.017 | 0.133 |
| B: tree, 6 aggregated sensors | 6 | 0.085 | 0.130 |
| **C: tree, forward selection (final)** | **3** | **0.0005** | **0.118** |
| D: tree, LASSO selection | 14 | 0.017 | 0.133 |
| *D2: tree, LASSO path chosen by regret (new)* | *3* | *0.0005* | *0.126* |
| *E: L1 logistic regression (new)* | *16* | *0.0024* | *0.126* |

The protocol is good: time-based split, test used once, choices frozen before the test, and a metric that matches the business problem.

---

## 6. Why the LASSO candidate (D) does not work

**What you see:** D has *exactly* the same scores as A on validation and on test. The LASSO step changed nothing.

**Why, in four points:**

1. **The tree never used the two sensors the LASSO removed.** The LASSO drops only `S1_Temp` and `S1_Sound`. Tree A only splits on `S1_Light`, `S2_Light`, `S3_Light`, `S2_Temp` and `S5_CO2`, all of which are still in D's 14 sensors. So trees A and D are the same tree.
2. **λ is chosen on a flat curve.** From 16 down to 11 sensors, the validation MSE stays within about 3% of its minimum. The chosen λ = 0.00037 (14 sensors) is one point on a plateau, so 11, 14 or 16 sensors are equally justified by this criterion.
3. **Only the 4 light sensors matter to the LASSO.** At the chosen λ, their coefficients are 0.29–0.59 (standardised). The 10 other kept sensors are all below 0.035. They are kept because each one lowers the MSE a tiny bit, and minimum-error λ is known to keep too many variables.
4. **The criterion does not match the goal.** The LASSO picks sensors that help a *linear formula* predict the count (MSE). The tree needs sensors that help it *choose the right ventilation level* (regret). These are different questions. The figure in 11.2 of the companion notebook shows it: the MSE keeps falling until about 11 sensors, while the tree's regret is lowest with 3–4.

**The fix (candidate D2).** Keep the LASSO, but use its path only as a *ranking* of sensors (the order they enter as λ decreases). For each set on the path, tune a tree and choose the set with the lowest **validation regret**, with fewest sensors as the tie-break. This is exactly the rule already used for candidate C, so C and D2 become directly comparable. Result: **3 sensors** (`S1_Light`, `S2_Light`, `S3_Light`), validation regret 0.0005 (tied with C), test regret 0.126 instead of 0.133.

**What I tried that does not fix it:** a 1-standard-error rule with 5 contiguous folds on the training period still keeps 10 sensors. The problem is the criterion, not the noise in λ. This is not in the companion notebook.

**An alternative (candidate E).** An L1-penalised **logistic** regression is a real 4-class classifier with real probabilities, and the L1 penalty also selects sensors. It does well (validation regret 0.0024), but at its best penalty it keeps all 16 sensors. It is a good model but not a good selector.

---

## 7. How the analysis can be completed: the proposed cells

All cells are in [proposed_cells.ipynb](proposed_cells.ipynb), already executed so you can read the outputs. **To use them:** paste every cell *after the first one* at the **end** of the main notebook, as a new section 11. The first cell (SETUP) only copies existing code so the file runs on its own; do not paste it.

They go at the end rather than inside section 8 because D2 and E were designed after the test results were seen. Putting them before section 9 would make the test no longer blind for them. The final choice (C) does not change.

| Section | What it adds | Key finding |
|---|---|---|
| 11.1 | Diagnosis of D | Flat MSE curve, tiny coefficients, the tree ignores the removed sensors, so D = A |
| 11.2 | **Candidate D2**: LASSO path + regret, with a two-panel figure (MSE vs regret by number of sensors) | 3 light sensors, tied with C on validation |
| 11.3 | **Candidate E**: L1 logistic regression | Validation regret 0.0024, keeps 16 sensors |
| 11.4 | Validation and test tables with D2 and E | D2 beats the old D on test (0.126 vs 0.133) with 3 sensors instead of 14 |
| 11.5 | **Error analysis** of the test failure + figure of 10 Jan + two "what if" fixes | 86–97% of every candidate's test regret comes from one 95-minute episode. Lights ≈ 7 lux, but CO₂ slope and motion detectors clearly show people. No model ventilates even once in this episode. |
| 11.6 | **Block bootstrap** (1-hour blocks) on test regret | Only 3 of 44 test hours have people, and every candidate differs from C in just one hour. The 95% intervals (≈ 0 to 0.3) overlap: **the test cannot rank the models.** |
| 11.7 | **Sensitivity to the cost ratio** (1.25 to 10) + whether the cost rule changes any decision | The ranking is stable for every ratio. Most tree leaves are pure (probability 1 on 89–100% of rows for A, B, C and D), so for the trees the expected-cost rule gives the same answer as "most likely count" on every validation row at the chosen costs. The decision layer only matters with real probabilities (E). |

**Two "what if" results from 11.5, to be discussed honestly as hindsight:**
- *Motion safety rule* (if a PIR fires, ventilate at least level 1): test regret 0.118 → 0.091, but validation regret 0.0005 → 0.0014. Our own protocol would have rejected it.
- *Tree without light sensors:* 0.15 on validation and 0.82 on test. The "empty room" values of some sensors drift between December and January: `S4_Sound` goes from 0.061 to 0.094 V, and the CO₂ slope is near 0 after days of an empty room instead of negative. A deep tree on these features does not transfer.

**Main conclusion to add to the write-up:** in this data, the light sensors make the problem look solved during the training and validation period, and nothing in that period can reveal the lights-off failure. More modelling cannot fix this; it needs more data: more days, other seasons, and people in the room with the lights off.

---

## 8. Smaller issues spotted in the main notebook's text

These are wording or number mismatches, not code bugs:

| Where | Issue |
|---|---|
| §8.2, text after candidate A's grid | Says "against **1.442** for 'never ventilate'", but the validation regret of never ventilate is **0.481** (table in §7). |
| §9 intro | Says choices are saved in `frozen_choices.json`; the code writes **`saved_choices.json`**. |
| §10, last text cell | "B comes back … even though on **training** data it was far behind": it was behind on **validation**. Also, "C stays the best" is not supported statistically (see 11.6). |
| §3, correlation paragraph | "Highly correlated variables will bias and break our model" is true for linear models (unstable coefficients, which is part of D's problem), but trees handle correlated inputs fine; they just pick one of them. |
| §5, "The decision y=a is always the best choice" | True only when y is known; worth saying "if we knew y". |
| Intro cell | Lists 6 steps, but the notebook has 10 sections. |

---

## 9. Practical notes

- The companion notebook was executed with Python 3.12, numpy 1.26, pandas 2.2 and scikit-learn 1.8 (not the pinned versions). It reproduces the main notebook's LASSO path and the validation and test scores of candidates A–D exactly. It rebuilds the four trees from `saved_choices.json` rather than re-running the forward selection.
- Candidate E uses `LogisticRegression(l1_ratio=1.0, solver="saga", random_state=0)`, the scikit-learn ≥ 1.8 way to ask for an L1 penalty. On an older version, write `penalty="l1"` instead (there is a comment in the cell). `random_state` is needed because `saga` shuffles the data; without it, E's scores change slightly from run to run.
- The companion notebook only *reads* `saved_choices.json` and never writes it.
