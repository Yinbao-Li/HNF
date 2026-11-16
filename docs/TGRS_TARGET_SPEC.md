# TGRS target spec — HNF v14 picking manuscript (agreed framing, 2026-11-16)

Purpose: define, before Phase 2 spending, (1) what IEEE TGRS actually
rewards, (2) what we want from a TGRS publication, and (3) the exact
promise this paper must keep. Every Phase 2/3 item is justified only
if it serves a line below.

## 1. What TGRS wants (editor + reviewer side)

TGRS is an engineering journal of the IEEE Geoscience and Remote
Sensing Society. Its bar is NOT a Nature-style conceptual breakthrough.
It rewards, in order:

1. In scope, framed as sensing. Theory/techniques for sensing the
Earth and processing/interpreting sensed data. For this paper:
seismic networks and OBS arrays as solid-Earth sensing
infrastructure; picking as inferring geophysical properties from
sensed waveforms. If page 1 reads as "a generic ML picker demoed
on seismic data", expect a scope rejection; EikoNet (TGRS 2021)
is the in-journal precedent to lean on.
2. A clear, single technical novelty that a competent engineer
can state in one sentence after reading the abstract.
3. Validation at the current community standard: strong recent
baselines, fair (matched-budget) comparisons, statistics that
survive scrutiny (seeds / CIs), honest failure analysis.
4. Usability by the community: code + weights + reproduction
path public at submission; efficiency numbers measured under a
stated, controlled protocol.
5. Formal compliance: ≤10 pages (overlength $230/page in 2026),
complete author/funding/conflict front matter, clean citations.

What TGRS does NOT require: SOTA on every benchmark, a new physical
law, or multi-domain generality. A method that trades some accuracy
for a verified, operationally valuable property (auditability) is a
legitimate TGRS contribution — IF the trade is quantified honestly.

## 2. What we want from TGRS (author side)

- A citable, peer-reviewed home for the HNF picking method and the
"verifiable decision structure" idea, in the GRSS community that
operates seismic/OBS sensing systems.
- Credibility collateral for the HNF programme: a public,
reproducible artifact (tgrs-v14) that later papers (incl. the
Nature-family facies/coda line) can cite as the established picker.
- NOT: the programme's flagship claim. The top-journal ambition
lives on the facies/coda line and starts after this is submitted.

## 3. The paper's one promise (must be keepable after Phase 2)

"The first seismic phase picker whose decisions are inspectable by
construction: a propagation-path proposal that a bounded CNN
refinement provably cannot override (Proposition 1, verified on
7,000 windows), with intervention evidence (memory-truncation
double dissociation) and factorized attribution showing exactly
what the physics prior does and does not buy — at competitive
accuracy and fully reproducible from a public release."

Every clause maps to existing or Phase-2 evidence. Any claim in the
manuscript that is not needed for this promise is a liability, not
an asset (e.g. universal "highest sub-0.1 s share" — false on
Ridgecrest; "36 scalars" — actually 45; Mars/template-matching —
discussion colour, move to supplement).

## 4. Reviewer-question → evidence map (Phase 2 justification)

- Is this in scope for TGRS? → Page-1 sensing framing + EikoNet
precedent → Phase 3
- Is the comparison fair? → Matched-budget EQT-aug/PN-aug (train to
~182 ep, eval at 20/40/80/182) → 2.2
- Where is a post-2022 baseline? → SeisT under STEAD-only protocol
(Path A); PickBlue 4C as labelled OBS ID reference (Path B) → 2.3
- Single run — luck? → 20-ep pair ablation ×3 seeds (Huygens vs
free); cluster-bootstrap CIs on headline + AE → 2.1 (reduced), 2.7
- Does the physics prior actually help? → Same pair ablation + AE
cluster CI; claim scoped to what survives → 2.1, 2.7
- What breaks it? → Low-SNR proposal failure: fix that preserves
Prop 1 (re-verify 0 created picks) or a dedicated scoped
subsection → 2.4
- Can I run it / how slow really? → tgrs-v14 release (done,
Phase 1); controlled re-bench + continuous throughput → 2.5
- ρ-scrub / waveform-edit claims? → Not claims in this manuscript —
supplement case studies or drop; do NOT scale up → 2.6 rejected

## 5. Definition of "ready to submit" (GO criteria for Phase 4)

- All §4 rows answered or the corresponding claim removed/downgraded.
- ≤10 pages; front matter complete; abstract contains only
claims rated SUPPORTED in the claim–evidence matrix.
- Public release reproduces ≥1 headline table end-to-end.
- Three-reviewer simulation returns no "reject" on procedural
grounds (scope, fairness, statistics, reproducibility).
