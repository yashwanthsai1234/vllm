# vLLM issue triage (local working notes, do not commit)

Snapshot: 2026-11-23, upstream `vllm-project/vllm`, local `main` at `6cbea92cb2`.
"Free" means: no assignee, no open PR that cites the issue, no claim in the comments.
Keyword PR searches were also run, but a related PR can still hide a fix. Re-check
before starting each issue (see "Before starting an issue").

---

## Plan

| Step | Issue | Area | Hardware | Status |
|---|---|---|---|---|
| 1 | [#54711](https://github.com/vllm-project/vllm/issues/54711) | KV cache (PP) | none | **Already fixed on main** by #60234 (2026-10-27): planner reads `group.layer_names`; regression test `test_kv_cache_config_skips_uniform_group_owned_by_other_pp_stage`. #61452 later made `layer_names` derive from `kv_cache_specs`. Open PR #63617 is stale. Action: comment on the issue so it can be closed. |
| 2 | [#67028](https://github.com/vllm-project/vllm/issues/67028) | KV offload | 2x L4 (any pre-Blackwell GPU) | Not started |
| 3 | [#69258](https://github.com/vllm-project/vllm/issues/69258) | torch.compile, GDN model | 1x H100 80 GB | Not started |

Backup for step 2: [#55766](https://github.com/vllm-project/vllm/issues/55766)
(KV + GDN model, reproduced on 1x RTX 3090).

Note: no free KV issue is fully CPU-only. #54711 may get a CPU unit-test repro.
#67028 needs a GPU, but a cheap one.

---

## KV-cache issues (free)

| Issue | Problem | Hardware | Notes |
|---|---|---|---|
| [#67028](https://github.com/vllm-project/vllm/issues/67028) | KV offload `canonical_layout` refuses to start with FlashInfer + block size 128 on pre-Blackwell GPUs | 2x L4, TP=2, Llama-3.2-1B, fp8 KV | Found by CI (#61846). Exact command in issue. 0 comments. |
| [#55766](https://github.com/vllm-project/vllm/issues/55766) | Qwen3.5/3.8 (GDN) all-NaN logits after a prefix-cache hit when the previous prefill ended 4-10 tokens past a block boundary | Reporter 2x H100 (Qwen3.8-27B); also 1x RTX 3090 (Qwen3.5-4B + MTP) | 30 s repro in issue. Comments say only the V1 model runner fails now. Many GDN PRs exist: check for overlap. |
| [#54711](https://github.com/vllm-project/vllm/issues/54711) | With PP, a rank that owns no layer of a KV-cache group still emits `KVCacheTensor`s; `allocate_kv_cache` dies with a bare `StopIteration` | Reporter PP=4 on 8x A100-class (hybrid model) | 0 comments. CPU unit-test repro is a guess, not yet checked. |
| [#66193](https://github.com/vllm-project/vllm/issues/66193) | Extensible KV cache counts the one-time FlashInfer autotune peak as serving memory; KV cache shrinks 2.4x | 1x RTX 5090 / RTX PRO 6000 (Blackwell) | Regression from #56492. Check #54209, #49115. |
| [#54926](https://github.com/vllm-project/vllm/issues/54926) | Gemma 4 MTP + NIXL P/D: draft model KV is not transferred | 2 GPUs + NIXL | #68347 may overlap. |
| [#64584](https://github.com/vllm-project/vllm/issues/64584) | DeepSeek-V4.1 decode logprobs drift after a prefix-cache hit with `swa_bounded_replay` | 8x B200, TP=8 | Expensive. |
| [#69154](https://github.com/vllm-project/vllm/issues/69154) | GLM-5.3-Flash + MTP + prefix caching: "locklock..." output, MTP acceptance drops to 0 | 4x B200, TP=4, FP8 | Intermittent, no repro script. |

### KV-cache issues: claimed, fixed or not a bug

| Issue | Status |
|---|---|
| #69137 | Claimed (@n33ll) |
| #68209 | Claimed (@tejachavali) |
| #62510 | Placeholder is intentional |
| #61532 | Claimed (@jiuyueQ) |
| #58394 | Covered by #58121 |
| #56828, #56823 | Mostly fixed on main (#51504) |
| #55951 | Fixed by #55506 |
| #49449 | Duplicate of #49377 |
| #49261 | Claimed (@yjlhhh) |
| #44250 | Fixed in LMCache (LMCache#2962) |
| #58034 | Claimed (@HritickGokul) |
| #58894 | PR #55601 |
| #52583 | Fixed by #51603 |
| #50454 | PR #50344 |
| #60350 | PR #60673 |
| #54035 | Fix in flash-attention fork PR 184 |
| #53982 | Patch circulating in comments |
| #65973 | Claimed (@Mist-wu) |
| #63693 | Reproduced without KV handoff, so not a KV bug |
| #51250, #51198, #53749, #54458, #53670 | Design limits of prefix caching on hybrid models, under discussion |
| #61160, #37729 | Large ongoing investigations, many people |

---

## GPU issues (free)

| Issue | Problem | Hardware | Notes |
|---|---|---|---|
| [#69258](https://github.com/vllm-project/vllm/issues/69258) | Qwen3-Next-80B next-token top-10 differs between `splitting_ops=None` and `splitting_ops=[]` under Inductor | Reporter 1x RTX A6000 48 GB + 120 GiB CPU offload; 1x H100 80 GB should work | Hard: may be expected compiler rounding. First prove which. 0 comments. |
| [#63646](https://github.com/vllm-project/vllm/issues/63646) | Qwen3.6-27B TP=8 crashes with "CUDA error: unspecified launch failure" on Opencode requests | 8x A100 40 GB | Vague; v0.32.0, may be fixed on main. 0 comments. |
| [#64005](https://github.com/vllm-project/vllm/issues/64005) | V2 model runner: Qwen3-32B TP=2 emits only token ID 0 | 2 GPUs | Probably stale (v0.28.0); not reproduced on 2x B200. |

### GPU issues: claimed in comments, no PR yet

| Issue | Problem | Hardware | Status |
|---|---|---|---|
| #68412 | `causal_conv1d_fn` writes outside state tensor for `pad_slot_id=-1` (Triton) | Any CUDA GPU | Two claimants, one has a fix |
| #68411 | Triton top-p keeps the whole row when normalization underflows | Any CUDA GPU | Likely covered by #66792 |
| #68669 | HiSparse on P/D decode fails for hybrid models (GLM-5.3-Flash) | 8x H20 | Claimed (@ssghost) |
| #64489 | Unlimited-OCR Docker image cannot serve its model | 1 GPU | Two contributors on it |

### GPU issues: resolved, duplicate or not vLLM

| Issue | Model | Status |
|---|---|---|
| #69320 | OmniASR | Fixed in Transformers (huggingface/transformers#51100) |
| #66250 | phi-2 | Not reproduced on current main |
| #65807 | Qwen3.8-27B AWQ | User config (`--generation-config vllm`) |
| #65337 | Qwen3.5 / Qwen3-Next family (GDN) | Reporter agreed the failing shape cannot happen |
| #63276 | DeepSeek V4.1 Flash | Probably the model; reporter moved to SGLang |
| #62803 | Custom checkpoint on Qwen3.5 | Unsupported model, native support coming |
| #68002 | ALiBi models | Duplicate of #67772 (PR #67773) |

Model note: Qwen3.5 (`qwen3_5.py`) and Qwen3-Next (`qwen3_next.py`) both use GDN
(Gated DeltaNet) layers. #69258, #55766 and #65337 are all GDN models.

---

## CPU-friendly bugs (all taken on 2026-11-23)

| Issue | Open PR |
|---|---|
| #69257 bench serve probe counts | #69262 |
| #68424 stop strings with no tokenizer | #69462 |
| #68324 detokenizer drops trailing U+FFFD | #68358 |
| #68579 duplicate tool names | #68588 |
| #68137 Qwen3 parser trailing whitespace | #68231 |
| #69267 Inkling streaming re-parse | #69297 |
| #69139 scheduler over-admits | #69316 |
| #68435 inactive ECTransferConfig | #68442 |
| #68434 lm-format-enforcer import | #68825 |
| #68427 compressed-tensors metadata | #68455 |
| #68610 num_requests_waiting metric | #68700 |
| #69210 fused collective benchmark | #69219 |
| #68595 /resume while asleep | #68602 |
| #68316 OOM hint | #68531, #68439 |
| #64940 Responses empty content | #49130 |
| #63465 Mooncake split-load | #63464 |
| #68779 Mamba + DFlash split | #54076 (waiting on rebase) |

Not actionable: #63310 (fixed by #35508), #64034 (closing), #68003 (duplicate),
#63794 (not vLLM), #65173 (works as designed), #69459 (blocked on #56418, #67924),
#68343 (fixed by #60856), #65644 (fixed by #55708), #67999 (#67790 fixed by #67791).

---

## Before starting an issue

```bash
gh issue view <N> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<N> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<area keywords>"
```

Then comment on the issue to claim it (you post this, not the agent).

## Git workflow (you run these)

```bash
git remote add upstream https://github.com/vllm-project/vllm.git   # once
git fetch upstream
git checkout -b fix/<N> upstream/main
# work, test, commit (Signed-off-by + Co-authored-by trailers)
git push origin fix/<N>
gh pr create --repo vllm-project/vllm --head yashwanthsai1234:fix/<N>
```

## PR requirements (vLLM AGENTS.md)

- Why it is not a duplicate of an existing PR.
- Exact pytest command and summary line (with time).
- Model eval results if output, accuracy or serving changes.
- Statement that AI assistance was used.
- Docs updated for user-visible changes.
- Run the `pr-checklist` skill before opening the PR.
- You review and can defend every changed line.

## Local setup still needed

- No `.venv` yet. Create with `uv venv --python 3.14`, then
  `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
  (check that precompiled wheels work on macOS arm64; a CPU build may be needed).
