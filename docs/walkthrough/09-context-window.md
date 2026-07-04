# 09 — The Context Window: size, RoPE limits, and what it means for finance

This doc answers three linked questions about ChronoGPT's **1792-token context window**:

1. **How big is it, really?** — in words, and in units a finance researcher cares about (a news article? a transcript? a 10-K?).
2. **What is the hard limit baked into the rotary embeddings, and does "doubling the context" break it?** — a precise look at the `1/1024` frequency ladder and the "cross 2π" intuition (which is *directionally* right but not the actual failure mode).
3. **How would you extend it, and what's the catch for *this* project specifically?**

Read `03-model.md` (RoPE section + Addendum II) first for the rotation mechanics; this doc builds on them.

---

## 1. How big is 1792 tokens?

The trained context is `block_size: 1792` (`configs/train.yaml:16`). In GPT-2 BPE, English runs ~**0.75 words/token** (~4 characters/token), and financial text — dense with numbers, tickers, punctuation, and dollar signs — tokenizes *less* efficiently than prose. So:

$$1792 \text{ tokens} \approx 1{,}300\text{–}1{,}400 \text{ words} \approx 2\text{–}3 \text{ double-spaced pages}.$$

Put in finance-document terms:

| Document type | Typical length | Fits in 1792? |
|---|---|---|
| Headline / StockTwits / tweet | 5–40 words | ✅ trivially |
| News article (Reuters/Bloomberg) | 300–800 words | ✅ comfortably |
| Analyst-report abstract / press release | 200–1,000 words | ✅ |
| FOMC **statement** | ~500–1,000 words | ✅ |
| 8-K item text | few hundred–few thousand | ✅ usually / 🟠 sometimes |
| FOMC **minutes** | ~5,000–8,000 words | 🔴 needs ~8–12k tokens |
| Earnings-call **transcript** | 5,000–15,000 words | 🔴 needs ~8–20k tokens |
| MD&A section (of a 10-K) | 5,000–20,000 words | 🔴 |
| Full **10-K / 10-Q / prospectus** | 30,000–100,000+ words | 🔴 40k–150k tokens |

**The takeaway for research design.** 1792 is *well-matched* to the classic "text-as-data" asset-pricing setup (Tetlock; Ke–Kelly–Xiu; Manela–Moreira), where signal is extracted **per article / per headline** and then aggregated to a firm-day or firm-month panel. There, 1792 is usually *more* than you need — a single article rarely exceeds it.

The window bites in exactly one regime: **document-level embedding of long filings or transcripts**. If your unit of analysis is "embed this entire 10-K / this whole earnings call and predict returns," you cannot feed it whole. You must either (a) **chunk** the document into ≤1792-token windows and pool the chunk embeddings (mean/attention pooling — see `06-infer-and-eval.md` on `embed(pool=...)`), or (b) **extend** the context (Section 3). Most of the ChronoGPT/ChronoBERT literature takes route (a); route (b) is a research project in itself.

---

## 2. The rotary limit, and the "cross 2π" question

### 2a. The frequency ladder

`Rotary` is built with `dim = head_dim = 128` (`model.py:72`), so `dim//4 = 32`:

```python
angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=32)   # 32 geometric freqs, 1.0 → 1/1024
angular_freq = cat([angular_freq, zeros(32)])                  # + 32 zeros → 64 entries
```

This gives **32 real rotation frequencies**, geometrically spaced from `1.0` down to `1/1024 ≈ 0.000977` rad/position (plus 32 zero-frequency "content" channels that never rotate). Each frequency `ω` turns a 2-D coordinate pair at angle `ω · position`; its **wavelength** — positions per full 2π turn — is `2π/ω`:

| Channel `i` | frequency `ω = 1024^(−i/31)` | wavelength `2π/ω` (positions) |
|---|---|---|
| 0 (fastest) | 1.000 | **6.3** |
| 8 | 0.167 | 38 |
| 16 | 0.0279 | 225 |
| 24 | 0.00468 | 1,344 |
| 31 (slowest) | 0.000977 = 1/1024 | **6,434** |

The fastest channel wraps every ~6 positions (highly periodic — that's *fine*, it resolves fine-grained local offsets); the slowest wraps once every ~6,434 positions and carries the long-range signal.

### 2b. Verifying your intuition — and correcting it

Your claim was: *"1/1024 has an upper limit where soon (when doubling the context) we'd face the same positional embeddings, since we cross 2π."* Two parts, and the arithmetic sharpens both.

**Part 1 — the number. Doubling does *not* cross 2π; it takes ~3.6×.** The slowest channel's angle is `pos × (1/1024)`:

- At the trained context **1792**: angle `= 1792/1024 = 1.75 rad ≈ 100°` — only **28% of one full turn**. The design leaves the slowest channel deliberately *un*-wrapped across the whole trained window (good design: no aliasing within training).
- **Doubling to 3584**: angle `= 3.50 rad ≈ 200°` — still **under 2π** (6.28).
- **Crossing 2π** needs `pos = 2π × 1024 ≈ 6434` — i.e. **~3.6× the current window**, not 2×.

So the periodicity ceiling is real, but it's at ~6,400 positions, not at "double." (The base here is `1024`, much smaller than the usual RoPE base of `10000`; a smaller base packs the wavelengths tighter, giving *less* long-range headroom — which is exactly why your "upper limit soon" instinct is qualitatively correct, just off by a factor.)

**Part 2 — the mechanism. Crossing 2π does *not* create "the same positional embedding."** This is the part worth correcting, because it's the crux of why context extension is hard.

- RoPE encodes **relative** position: the attention score depends only on the offset, `⟨R_m q, R_n k⟩ = g(n − m)`. Per channel, that contribution is `cos(ω·Δ)`, periodic with period `2π/ω`. So on the *slowest* channel, two token-pairs whose relative distance differs by exactly ~6,434 get the **same contribution from that one channel**.
- But a full position/offset is encoded by **all 32 frequencies at once**, and they are mutually **incommensurate** (irrational ratios). For two *different* offsets to produce an identical embedding vector, *every* channel would have to realign simultaneously — which effectively never happens in any practical range. So true "collisions" are not the failure mode.

**What actually breaks under extension is _extrapolation_, not aliasing.** During training the model only ever saw relative offsets in `[0, 1792)`, so the slow channels only ever rotated through angles in `[0, 1.75]` rad. Feed it a sequence of length 4000 and those channels swing into angle regions **never seen in training** (`> 1.75` rad, out toward 2π and beyond). The attention logits in that regime are untrained and typically misbehave — they blow up or collapse, and quality falls off a cliff. That's the real ceiling: **out-of-distribution rotation angles**, which is why the fix is always to *rescale the angles back into the trained range* rather than to worry about 2π wraparound per se.

> One-line summary: your "upper limit" intuition is right; the number is ~3.6× (not 2×); and the failure is **extrapolation into untrained angles**, not identical embeddings from crossing 2π.

---

## 3. Extending the window (and the catch for this project)

All standard methods share one idea: **keep the rotation angles inside the range the model was trained on** by rescaling frequencies or positions. Three, in increasing sophistication:

- **Position Interpolation (PI; Chen et al. 2023).** Multiply every position by `L_train / L_target` before rotating, so a length-`L_target` sequence only ever uses angles in the trained `[0, L_train)` band. Dead simple; needs a short fine-tune to recover quality. Compresses fine-grained resolution uniformly.
- **NTK-aware scaling.** Instead of scaling all positions equally, change the **base** so high-frequency (local) channels are barely touched while low-frequency (global) channels stretch. Preserves local resolution better; can be applied nearly training-free for modest extensions.
- **YaRN (Peng et al. 2023).** NTK-by-parts + an attention-temperature correction; current best-in-class for large extensions with minimal fine-tuning.

**The catch specific to ChronoGPT replication.** Two constraints make extension more than a config flip here:

1. **Small base → little headroom.** With base `1024` (vs the usual `10000`), the slowest wavelength is only ~6,434 — the model has less "unused" low-frequency range to interpolate into than a typical `10000`-base model. Aggressive extension (to transcript length, ~16k) will lean hard on fine-tuning.
2. **The temporal screen still applies.** Extending context means fine-tuning on **long documents**, and every one of those documents must satisfy the no-look-ahead contract (`04-data.md` §6): it must be knowable before the vintage cutoff τ. You cannot just grab modern long-context finance corpora — you'd need long, *pre-cutoff* documents, which is a real data-collection constraint unique to this chronologically-consistent setting.

**Practical recommendation.** For the return-prediction use case, **don't extend** — chunk long filings/transcripts into ≤1792 windows and pool (route (a) in Section 1). Reserve context extension (PI or YaRN + a temporal-screen-compliant long-doc fine-tune) for a *separate* research question where whole-document reasoning in one pass is genuinely required, and budget it as its own project — not a preprocessing step.

---

## Appendix — where each number comes from

- `block_size = 1792` → `configs/train.yaml:16`.
- `head_dim = model_dim / num_heads = 1536 / 12 = 128` → `03-model.md`.
- `angular_freq = (1/1024) ** linspace(0, 1, dim//4)` with `dim//4 = 32` → `model.py:44`.
- Slowest frequency `1/1024`; wavelength `2π·1024 ≈ 6434`; angle at 1792 `= 1792/1024 = 1.75` rad — all direct arithmetic from the frequency formula.
- Words-per-token ≈ 0.75 (GPT-2 BPE, English); lower for numeric finance text.
