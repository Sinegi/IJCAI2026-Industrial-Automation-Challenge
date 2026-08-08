# Technical report — Industrial Automation Challenge, Track 1
## 1. Task and system

The competition set contains 3,048 four-to-eight-option questions: 2,758
*relational* items asking which sensor signal indicates a given failure mode (or
the reverse), and 290 *general* industrial-engineering items. Our system is a
single open-weight language model, `Qwen/Qwen3-8B` (8.19 B parameters, 6.95 B
non-embedding), adapted with QLoRA. No retrieval, tool call, external API or
second model takes part in inference: the model sees the passage, the question
and the options, and nothing else.

## 2. Answer decoding: constrained logits rather than generated text

The supervised target ends at `<think>…</think>\n<answer>`, immediately followed
by the end-of-sequence token. The model is therefore never trained to emit the
option letter as text.

Supervision is applied to the logits at the single position that follows
`<answer>`. Those logits are gathered down to the option letters the item
actually offers, and a **K-way softmax cross-entropy** is taken over that
restricted set, with label smoothing 0.05 distributed only over the same row's
candidates. Because the softmax normalises across exactly the options present,
the objective is invariant to the option count, which our augmentation varies
between epochs.

Decoding mirrors the objective exactly: generate the reasoning trace, run one
forward pass over `prompt + trace + "\n<answer>"`, restrict the final-position
logits to that item's letters, take the argmax. Training and inference operate
on the same tensor position, an option outside the presented set is
unrepresentable, and no probability mass escapes to the remaining ~150k
vocabulary entries.

One consequence shaped the reasoning targets: the target trace **never states
the answer**. If it did, teacher forcing would place the answer inside the
model's own context and the letter objective would degenerate into copying it —
while at inference the trace is generated, so a faulty trace would be copied
just as faithfully.

## 3. Input and target visibility

The evaluation items carry only a passage, a question and options. Asset class,
relation direction, question polarity and anchor entity exist in our training
data but are withheld from both the prompt and the target text, and are used
only as side information for data construction. A target that named them would
train the model to invent, at test time, fields that do not exist. The prompt
contains no metadata header and no entity markers.

The instruction paragraph is the only branch: relational items receive an FMEA
instruction naming the failure-mode → physical-quantity → sensor-response chain,
general items receive one that does not. The branch reads the `question_type`
field supplied with the data — no classifier and no heuristic.

## 4. Training-data construction I: relation graph

From the training split we induce a bipartite graph over failure modes and
sensors per asset class, recording **observed non-edges** separately from pairs
that were simply never observed. Only the former are used as distractors:
"absent from the graph" and "known to be unrelated" are different claims, and
conflating them manufactures false negatives.

The graph then drives item synthesis rather than the loss:

- **Template enumeration** — instead of only resampling existing items, we
  enumerate the question shapes the graph can support across option counts 4–8,
  covering shapes present in evaluation but rare in training.
- **Distractor resizing** — items are regenerated every epoch at option counts
  2, 3, 4, 6 and 8, shrinking as well as growing. Error analysis showed the gold
  option was ranked second in most residual errors, i.e. the final two-way
  discrimination was the untrained regime.
- **Cross-asset distractors** — entities that are genuine neighbours of the
  anchor in a *different* asset class but absent from this one (30% of added
  distractors), which penalises answering by lexical familiarity.
- **Coverage floor** — a minimum of 12 items per relation fact (edge × direction).
  Coverage was heavily skewed: a long tail of facts appeared only once or twice
  while common ones appeared twenty times or more, and a fact seen twice in a
  large corpus is not learned.
- **None-of-the-above (NOTA) items** — both items whose correct answer is NOTA
  and items where NOTA is present but wrong, so the model learns the distinction
  rather than a prior over the phrase.
- **Per-epoch option permutation** — option order is reshuffled in place each
  epoch rather than materialising permuted duplicates.

Three explicit shortcut controls: a length band of 0.55 between gold and
distractor lengths, so answer length carries no signal; a difficulty mix of
30/45/25 (easy/medium/hard), because training only on maximally hard distractors
makes "least similar option wins" a viable rule without any relational
knowledge; and a cap of 3.0 on how often a fact may appear as a distractor
relative to appearances as the gold answer, since the letter objective otherwise
suppresses facts that are structurally over-represented in distractor slots.

## 5. Training-data construction II: knowledge grounding from public documents

Graph-derived synthesis alone left two gaps. The 290 general items are not
relational at all, and error analysis of an earlier checkpoint found that most
validation errors were *over-abstention*: NOTA selected although a correct
option was present. That is a narrow corpus collapsing into graph lookup —
an unobserved pair is read as "no relation exists" rather than "not recorded".

We therefore grounded the model in **17 publicly available technical documents**
from U.S. government agencies: DOE sourcebooks and best-practice guides (pumps,
fans, motors, steam, compressed air, internal-combustion engines, operations and
maintenance), EPA procedures (compliance assurance monitoring, flow
measurement), FAA airframe and powerplant handbook chapters, a NETL gas-turbine
handbook, a NIST technical note, and OSHA guidance (electrical safety,
lockout/tagout, job hazard analysis). All are free to access and carry no usage
restriction.

Provenance is recorded per source in a manifest: identifier, title, agency,
publication date, URL, SHA-256 digest, byte size, the specific pages used, the
number of facts derived, and the retrieval date — so any generated item can be
traced to a document and page.

From these we extracted **662 structured facts**: 329 mechanism statements, 186
reviewed items, 110 fault statements and 37 maintenance-decision statements.
The distractor structure is carried by the extraction itself rather than added
afterwards: each fault fact records an opposite, a no-effect, an over-general
variant and at least two rival mechanisms; reviewed facts record four to six
confusable concepts; decision facts record four ranked candidates. These are
same-domain, same-register distractors, so the resulting items cannot be solved
by word association.

Roughly half of the extracted facts mention an asset, sensor or failure mode
that also appears in the relation graph, so the document items reinforce the
relational subset rather than forming a disconnected second task.

Generation per fact: four positive items, three negatively-phrased items, 25%
*affirmation* items — where a correct option is present and the model must not
abstain, targeting the over-abstention failure directly — 10% items whose
correct answer genuinely is NOTA, and 30% surface-form variation so a concept is
not tied to a single string.

## 6. Weight averaging across seeds

The final adapter is a weight-space average ("model soup") of three runs of the
identical recipe at seeds 42, 1234 and 2025.

What must be averaged is the **product** `B·A` of each low-rank pair, not `A`
and `B` separately: the factorisation is not unique, so `mean(A)·mean(B)` is not
the mean of the updates. Stacking the `A` factors vertically, the `B` factors
horizontally and scaling by `1/k` reproduces `mean(Bᵢ·Aᵢ)` exactly, at rank
`k·r`. This is why the released adapter has rank 192 while each constituent run
was trained at rank 64. Unlike an output-space ensemble it costs one adapter and
one forward pass.

## 7. Training configuration

| | |
|---|---|
| Quantisation | 4-bit NF4, double quantisation (QLoRA) |
| LoRA | rank 64 (192 after averaging), α 128 (384 after averaging), dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Schedule | 2 epochs, effective batch 64 (8 × grad-accum 2 × 4 DDP ranks) |
| Optimiser | paged AdamW 8-bit, lr 6e-5 cosine, warmup 10%, weight decay 0.05, grad-norm clip 0.5 |
| Sequence length | 768 |
| Regularisation | NEFTune embedding noise, α 1.0 → 10.0 |

**The loss has exactly two terms**: the standard language-modelling loss and the
constrained letter cross-entropy of §2, weighted 1.0. Auxiliary structural
objectives were implemented and evaluated during development — representation
alignment on graph edges and non-edges, an adjacency-conditioned matrix
regression, and a listwise ranking loss over option contents — and all are
disabled (weight 0) in the final recipe. The relation graph contributes through
data construction only, never as a training signal.

## 8. Results and limitations

On our held-out validation split the adapter answers **1,242 / 1,242 items
correctly (100.0%, bootstrap 95% CI [1.000, 1.000])**, with a median decision
margin of 0.949 between the top two option probabilities.

Two limitations should be read alongside that number.

**The validation split contains relational items only.** It has no counterpart
to the 290 general items in the evaluation set, so that portion of the task is
unmeasured by our own held-out data, and 100% describes the relational regime.

**A small number of evaluation items are effectively ties.** The decision margin
over the evaluation set has median 0.949 but minimum 0.000. Those items can flip
under numerical differences — notably a GPU without bfloat16 support, where the
4-bit compute path falls back to fp16. The agreement check in the agreement check in the notebook
quantifies this rather than leaving it to assumption.
