# Future TODOs for the LoongArch / NPYV port

Single source of truth for deferred work on the `loongson-experimental`
branch. Each section gives the goal, the approach, the rough cost, and
what success looks like — so a future session (or contributor) can pick
one up cleanly.

OPTIMIZATIONS.md's "Open follow-ups" section at the top of that file
points here; do not duplicate items there.

---

## Tier A — actively planned

### Vectorize the Payne-Hanek reduction for f64 sin/cos

The slow path (`|x| > 2^20`) in `npyv_sincos.h` currently runs scalar
Payne-Hanek per lane (see `payne_hanek_f64.h`). It is **correct across
the full f64 range** and benches at ~7-9 M/s — about ~25% under scalar
libm. The polynomial after the reduction is already vectorized; what
remains is to vectorize the reduction itself.

A naive vectorization (per-lane gather of 4 chunks of 2/π, vector FP
arithmetic for multiplication and accumulation) was attempted on
2026-05-24 and **fails on precision**: 1 ULP for `|x| ≤ 10^6`, 3000+
ULP for `|x| ≳ 10^10`. The root cause is precision loss during chunk
accumulation — partial products land at widely-separated exponents
(e.g. `+34`, `+10`, `−14`, `−38` for `x ≈ 10^10`), and summing them
in plain f64 drops ~30 bits at each step.

Three working approaches, in increasing order of engineering effort:

#### Variant A — Double-Double accumulation

Wrap every multiplication and addition in DD arithmetic (Dekker-style
two-sum with FMA correction). Each chunk product becomes a `(hi, lo)`
pair; each accumulator step uses `ddadd2` so the lower-order bits
survive.

- **Files:** `npyv_sincos.h` gets a new `npyv__payne_hanek_simd_reduce`
  plus inline `ddadd2` / `ddmul` helpers. `npyv__sincos_f64_slow_path`
  delegates to it for finite inputs.
- **Cost:** ~100-150 lines of careful C plus ULP validation.
- **Expected speedup:** ~3× slow path (7 → ~20 M/s).
- **Best for:** quick win; cleanest small change.

#### Variant B — Multi-precision `iq[]` style

Port FDLIBM's `__kernel_rem_pio2` directly: keep the running sum as N
separate 24-bit-piece vectors (matching scalar `iq[0..jz]`), propagate
carries vector-wise. Each "piece" is one `npyv_s64` vector across lanes.
The distillation loop and the `ih > 0` complement-and-borrow steps
become straight-line vector ops over those piece-vectors.

- **Files:** new `npyv_payne_hanek_simd.h` with the multi-precision
  SIMD reduce; thread into `npyv__sincos_f64_slow_path`.
- **Cost:** 250-300 lines (closest port of the scalar FDLIBM algorithm).
- **Expected speedup:** ~4× slow path (7 → ~30 M/s).
- **Best for:** most faithful to the proven algorithm; lowest precision
  risk because it mirrors what the scalar version already does correctly.

~~Variants A/B/C all DONE via Variant C below — 2026-05-24, commit 41200fefd0.~~

#### Variant C — SLEEF-style pre-aligned table (LANDED)

Generate a precomputed table where each entry already holds 4 doubles
at the right magnitude for direct multiplication with `x`. Drop the
per-chunk scaling entirely; reduction becomes "gather 4 doubles per
lane, multiply, sum with DD".

- **Files:**
  - `numpy/_core/src/umath/payne_hanek_table.py` (generator, partial —
    written 2026-05-24, see commit notes; needs the missing quadrant
    extraction step).
  - new `numpy/_core/src/umath/payne_hanek_simd_table.h` (the generated
    table — 4 doubles × ~1024 entries ≈ 32 KB)
  - update `npyv_sincos.h` to use it.
- **Cost:** ~200 lines of C plus ~50 lines of Python. Table grows from
  264 bytes (current 24-bit-chunk table) to ~32 KB.
- **Expected speedup:** ~4-5× slow path (7 → ~35 M/s).
- **Best for:** the end-state if we ever want to propose the LoongArch
  sin/cos path upstream. SLEEF's structure is well-trodden.

**Status (2026-05-24):** WIP — got the residual-encoded table working
bit-exact (`dd_sum = x · 2/π / 2^(e_x − 1)` exactly for all tested
e_x up to 2^1000). The remaining gap is extracting
`(x · 2/π) mod 4` from that DD sum: bits at positions
`−(e_x − 1), −(e_x − 2)` of the DD value give the quadrant, but DD
has ~106 bits of precision so this only works for `e_x ≤ ~106`.

Three concrete paths to close the gap:

  1. **Separate quadrant path.** A small scalar (or SIMD-integer)
     routine that extracts just the integer-mod-4 from x's bits and a
     2/π bit-window table, while the DD chunks here provide the
     fractional reduced argument. Splits the problem cleanly.
  2. **More chunks per entry.** Store 6 or 7 doubles per entry instead
     of 4, giving ~150 bits of DD precision. Then the mod-4 extraction
     works for e_x up to ~150. Still doesn't cover full f64 range but
     would cover a huge majority of real workloads.
  3. **Find and port SLEEF's actual gen script.** Their f64 path
     handles the full range somehow — the trick they use isn't visible
     from just reading `rempitabdp[]` values (the negative residuals
     hint at the DD-encoding structure but the quadrant story isn't in
     the obvious code path).

The generator file `payne_hanek_table.py` is committed as a starting
point with the working-but-incomplete formulation. The header it would
emit (`payne_hanek_simd_table.h`) isn't checked in until the missing
piece is resolved.

Pick **Variant B** if the goal is correctness-first; Variant A if the
goal is "ship something faster quickly"; Variant C if the goal is to
converge on what other portable SIMD math libraries already do.

The current scalar PH is correct, ships, and passes all 4699 umath
tests. The bench is unaffected (all bench inputs are fast-path), so
any of these variants is purely a slow-path win.

### Vectorize f16 transcendentals via an f16↔f32 SIMD bridge

`loops_half.dispatch.c.src` has an SVML-only SIMD path; everything
else (including LoongArch) falls through to a scalar `UNARY_LOOP { half
→ float → npy_<func>f → float → half }`. That is fully scalar — every
element does a half-to-float convert, a scalar libm call, and a
float-to-half convert.

The opportunity: LASX provides f16↔f32 lane conversions
(`xvfcvtl_s_h` / `xvfcvth_s_h` for f16→f32, `xvfcvt_h_s` for f32→f16).
Wire those into a small "bridge" kernel that loads 16 f16 values,
splits into two 8-wide f32 vectors, calls our existing f32 NPYV kernel
(already optimized for the 19 transcendentals via the Tier-1/Tier-2
work), packs back to f16, stores.

- **Files:** new `numpy/_core/src/umath/npyv_f16_bridge.h` (or extend
  `loops_half.dispatch.c.src` with a LoongArch branch).
- **Cost:** ~80 lines for the bridge plus one-line wiring per ufunc.
- **Expected speedup:** 4-15× per f16 op vs current scalar libm path
  (depending on how much SIMD parallelism the f32 kernel already has).
- **Validates against:** existing f16 test_umath cases; ULP tolerance
  should match the f32 kernel since the only extra error is the f16
  rounding at the output (≤ 1 ULP of f16).
- **Covers ops:** sin, cos, tan, exp, exp2, expm1, log, log2, log10,
  log1p, cbrt, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh,
  arccosh, arctanh — i.e., everything `loops_half.dispatch.c.src`
  currently scalarizes on non-x86.

### Integer SIMD — survey results and remaining work

Survey done 2026-05-24. Initial findings:

  - `loops_comparison`, `loops_minmax`: shipped LSX but no LASX.
    Adding LASX gave 2-5× wins on int8/int16 ops (4.8× for `int8`
    add, 2.7× for `int16` add, 1.3-2× across the integer min/max
    reductions). int32/int64 gains were modest because LSX already
    saturates the 3A6000's per-lane integer pipeline. **Landed in
    commit 77db9e0f4e.**

  - `loops_arithmetic`: **LASX excluded pending bug fix.** When LASX
    was added to its dispatch list, `test_signed_division_overflow[h]`
    in `test_umath.py` started missing the int16 overflow warning.
    The kernel `simd_divide_by_scalar_contig_s16` does
    `npyv_tobits_b16(npyv_not_b16(noverflow))` to detect any lane
    that hit `INT16_MIN`. On LASX, `npyv_tobits_b16` (in
    `numpy/_core/src/common/simd/lasx/conversion.h`) does an
    `xvpickev_b` + `xvmsknz_b` dance and re-assembles a 16-bit mask
    from two 8-bit halves; one of those halves appears to drop bits
    for the INT16_MIN pattern. LSX dispatch is unaffected. **Fix:**
    audit the LASX `npyv_tobits_b16` against the LSX version, write
    a focused unit test, then re-enable LASX in the dispatch list.
    Estimated 1-2 hours.

  - ~~`loops_modulo`: LSX/LASX kernel~~ — landed 2026-05-24 in commit
    e6acbf145b. `np.mod` / `np.remainder` / `np.fmod` / `np.divmod`
    all benchmark 3–15× faster on int{8,16,32,64} and uint{8,16,32,64}
    (best: int8 mod 165 → 2471 M/s = 15×). Uses native LASX/LSX
    `xvdiv_*` and `xvmod_*` intrinsics — no VSX4-style 8→32 expansion
    needed.
  - **Open follow-up:** `np.floor_divide` still hits the scalar
    BINARY_LOOP because it dispatches through
    `loops_arithmetic.dispatch.c.src`, which only has a
    divide-by-scalar kernel (no vector-by-vector). Adding a
    LoongArch vector-by-vector floor_divide kernel parallel to the
    new loops_modulo one would give the same 3–15× boost on
    `np.floor_divide(a, b)`. Estimated ~150 lines.

  - Bitwise ops, sum/argmax reductions: already fast on LSX
    (gigaelements/sec). LASX adds modest ~10-30% beyond LSX. Low
    priority.

  - Highway 16-bit qsort regression on LASX: still on the list (see
    Tier B below).

---

## Tier B — smaller follow-ups

### arcsinh f64 direct polynomial

Its `log1p` argument (`|x| + x²/(1+√(1+x²))`) is mostly > 0.5 for
typical inputs, so the existing log1p per-block Padé doesn't fire
through that call site. Would need arcsinh's own 4-range FDLIBM-style
reduction. Modest gain; deferred.

### cbrt f32 direct polynomial

f32 cbrt currently composes on f32 exp/log and benches at 3.57× over
main. Could be rewritten with the same FDLIBM bit-magic + Newton as
cbrt f64 for further gain. Lower priority because f32 has more SIMD
parallelism — the composition is already cheap.

### power f32 — lowest-speedup bench op

`power` f32 sits at 1.56× over main, the lowest in the bench table.
Investigate whether the special-case mask gymnastics (handling
`pow(0, y<0)`, negative-base integer-exponent, etc.) can be pruned
on the hot path while still preserving correctness, or whether the
underlying exp/log can be specialized for the `pow` use case.

### Highway 16-bit qsort regression on LASX

When LASX was added to the `highway_qsort` dispatch list, the 16-bit
sort regressed (94 → 59 Melem/s). LASX is currently excluded from
`highway_qsort_16bit`. Worth revisiting: maybe a threshold tweak or
Highway version bump fixes it.

### subnormal cbrt — actually, this one is DONE

cbrt f64 subnormal handling shipped 2026-05-23 (commit a244ab7aa7).
Listed here for completeness so future readers don't reopen the issue.

---

## Tier C — strategic / upstream

### Upstream proposal for the f64 sin/cos hybrid

Once one of the Payne-Hanek SIMD variants (A/B/C above) lands and the
precision story is "matches libm across full f64 domain via SIMD",
the sin/cos f64 hybrid in `loops_trigonometric.dispatch.cpp` is a
candidate for an upstream PR gated on a build flag (or a NPYV
feature check) rather than on `__loongarch__`. The numpy-discussion
thread that originally disabled SIMD f64 sin/cos cited precision
concerns with a now-retired SVML kernel; our Cody-Waite +
Payne-Hanek combo addresses that concern.

PR description should:
- Link to the original discussion thread.
- Show ULP comparison tables vs libm across the full domain.
- Bench both the fast-path range and the slow-path range.
- Be opt-in initially (build flag), opt-out later once it's seen
  field validation.

### Bench infra: asv with isolated venvs

`bench/loongarch.py` is a hand-rolled standalone microbench. Upstream
numpy uses `asv` (airspeed velocity). On this branch, asv was tried
on 2026-05-21 and failed because Python 3.13.13's `_multiprocessing`
lacks `SemLock` (issue gh-48020) and asv's isolated build process hit
ninja race conditions. If we want to upstream any of this, we'll need
the same data via asv. Either:
- Patch `asv/util.py:new_multiprocessing_lock` (we already did this
  locally; could be a small PR to asv).
- Or wait for a newer Python that exposes SemLock again on this
  build configuration.

### LSX-only fallback story

All optimization work targets LASX (256-bit). Older Loongson hardware
(3A5000) only has LSX (128-bit). Audit each kernel to confirm it
gracefully falls back to LSX when LASX isn't compiled in; verify with
`-mno-lasx` build. Most kernels are LASX-or-nothing today; some that
need only the polynomial (no f16 bridge or PH gather) work equally
well on LSX.
