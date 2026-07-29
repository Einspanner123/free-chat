# LLM Serving Optimization — Design Document

## 1. Problem Analysis

### 1.1 What makes LLM serving different from traditional API serving?

| Dimension | Traditional API | LLM Serving |
|-----------|----------------|-------------|
| Request duration | 1–50ms | 1–30s (one token at a time) |
| Response pattern | Single payload | Streaming: tokens arrive incrementally |
| GPU dependency | Rare | Essential; one request can saturate a GPU |
| Memory pattern | Transient objects | KV Cache persists across tokens, grows with sequence |
| Workload shape | Homogeneous request sizes | Heavy-tailed: some requests are 8 tokens, some are 8192 |

These differences mean that general-purpose API patterns (per-request worker pools, sync request-response) are fundamentally wrong for LLMs. The serving system must be aware of token-level iteration, GPU memory allocation, and request length heterogeneity.

### 1.2 The four resource bottlenecks every LLM deployment hits

```
Order of appearance as you go from prototype to production:

  1. GPU Memory     ← The first wall you hit
  2. Throughput     ← The second, when you have real traffic
  3. Latency (TTFT) ← Becomes visible with user-facing apps
  4. Cost           ← The meta-bottleneck that constrains all above
```

**GPU Memory** is the first bottleneck because:
- Model weights occupy VRAM before you serve a single request
- KV Cache grows linearly with `batch_size × sequence_length × num_layers × num_heads × head_dim`
- At 32K context with 7B model: ~16GB just for KV Cache (FP16)
- A 24GB RTX 3090 can fit the model OR the KV cache for long sequences, not both

**Throughput** becomes the bottleneck because:
- Autoregressive decoding is sequential at the token level
- Multi-token speculation trades GPU computation for parallelism
- Continuous batching trades scheduling complexity for GPU utilization

**TTFT (Time to First Token)** matters because users perceive it directly. A user staring at "..." for 3 seconds has a different experience from seeing the first word appear in 300ms.

**Cost** is the meta-constraint. Every optimization is ultimately a tradeoff between:
- Quality (accuracy of the model + accuracy of the served response)
- Speed (tokens per second, with and without batching)
- Memory (total VRAM for model + KV cache per request)
- Hardware cost (can you use consumer GPUs or do you need A100s?)


## 2. Architecture

### 2.1 Serving Pipeline

```
                          Incoming Requests
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Request Queue        │
                    │   (FIFO or prioritized)│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Scheduler            │
                    │                         │
                    │   Decides per-step:     │
                    │   - Which requests run  │
                    │   - Which are preempted │
                    │   - Which are swapped   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Memory Manager       │
                    │                         │
                    │   Owns:                 │
                    │   - KV Cache blocks     │
                    │   - Model weights       │
                    │   - Workspace buffers   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Model Executor        │
                    │                         │
                    │   Runs one step:        │
                    │   input → attention     │
                    │   → FFN → output token │
                    └───────────┬───────────┘
                                │
                                ▼
                          Output Token
                     (streamed to client)
```

### 2.2 Why this split?

The split between Scheduler, Memory Manager, and Model Executor mirrors how GPU time and GPU memory are managed:

- **Scheduler** decides *when* things run. It optimizes for throughput and fairness.
- **Memory Manager** decides *where* things live. It optimizes for VRAM utilization.
- **Model Executor** does the actual computation. It should be as thin as possible.

In a naive implementation, these three concerns are tangled: the model executor allocates its own KV cache in a single contiguous block, the scheduler assumes one-batch-fits-all, and there's no coordination. The result is that one long request starves short requests, and memory is wasted on padding.

The design constraint is: **the scheduler cannot make decisions without knowing the memory state, and the memory manager cannot allocate without knowing the scheduler's plans**. This requires them to be co-designed, not independent modules.

### 2.3 Interaction at Each Step

```
Scheduler step:
  1. Check: which running requests generated their EOS token?
  2. Free: tell Memory Manager to release KV blocks of finished requests
  3. Check: how many free blocks are available?
  4. Admit: pull from pending queue until blocks exhausted or batch limit hit
  5. Run: tell Model Executor to run one step on current batch
  6. Stream: send generated tokens back to clients
```

This loop runs 50–200 times per second (at ~50ms/step) for a typical 7B model. Every millisecond of overhead in this loop is a millisecond that the GPU is idle.


## 3. Component Design

### 3.1 KV Cache Manager

#### Why KV Cache management matters

Without KV Cache: every token generation step re-computes attention over all previous tokens. For a 4000-token sequence generating 100 new tokens, the `O(L²)` attention cost per step makes the total cost `O(L³)` — completely intractable.

With KV Cache (but naive management): the KV states for all previous tokens are stored and reused. The cost per step becomes `O(L)`, and total generation cost is `O(L²)`. This is the minimum for autoregressive decoding.

The problem is *what to do when the cache doesn't fit in memory*. At 32K context, a 7B model generates 16GB of KV cache entries. If you have 5 concurrent requests, that's 80GB — far exceeding any consumer GPU.

#### Design: Block-based Memory Pool

Rather than allocating one contiguous KV cache per request, the KV cache is divided into fixed-size blocks (analogous to OS page frames). Each request holds a linked list of blocks. When the cache needs to grow, a new block is allocated from the free pool. When a request finishes, all its blocks are returned.

```
Free Block Pool:   [▓][░][░][░][░][░][░][░]   (8 blocks total, 7 free)

Request A:         [▓]─→[ ]
Request B:         [▓]─→[▓]
Request C:                 [▓]

Eviction:  when free pool is empty and a new allocation is needed,
           choose a victim block based on policy, write it to CPU RAM
           if needed, and reassign to the requesting sequence.
```

**Block size tradeoff**:
- Small blocks (e.g., 16 tokens): finer memory utilization, more bookkeeping overhead
- Large blocks (e.g., 1024 tokens): less overhead, more internal fragmentation
- Practical choice: 256 tokens per block is a common default

#### Eviction Policy Design Space

Not all tokens are equally important for generation:

| Token Type | Importance | Evidence |
|-----------|------------|----------|
| Initial tokens (sink) | High | Attention scores concentrate here regardless of content; critical for positional encoding |
| Recent tokens | High | Short-term dependencies matter most for next-token prediction |
| Middle tokens | Variable | Some tokens are "heavy hitters" that aggregate attention; most are low-information |

**Policy comparison:**

| Policy | Memory | Algorithm | When to use |
|--------|--------|-----------|-------------|
| Full (no eviction) | 100% | Keep everything | Latency-critical, short sequences |
| LRU | 50–70% | Evict least recently accessed blocks | General purpose |
| Window | ~40% | Keep last W tokens + first S sink tokens | Long sequences, predictable bound |
| Attention-weighted | 30–50% | Compute attention scores during decode, evict lowest-scoring blocks | Best accuracy-memory tradeoff |
| StreamingLLM | 25–30% | Keep sink + recent, discard middle | Extreme memory pressure |

**Implementation note**: Attention-weighted eviction (e.g., H2O paper) requires computing attention scores during decoding. This adds ~5% overhead per step but provides the most intelligent eviction decisions. For a server that expects to handle many long-context requests, the overhead is justified by the memory savings.

#### Prefix Cache

When multiple requests share a common prefix (system prompt, few-shot examples, or a preamble), their KV states for that prefix are identical. Computing it once and reusing saves both GPU computation and memory.

```
Request A:  "You are a helpful assistant. | What is Python?"      ──┐
Request B:  "You are a helpful assistant. | Explain gravity."      ──┤
Request C:  "You are a helpful assistant. | Write a haiku."       ──┘
                                │
                     Shared prefix: KV states cached once
                     Divergent suffix: KV states per-request
```

The prefix cache is keyed by a hash of the prefix text. On a cache hit, the KV blocks are copied (reference-counted) rather than recomputed. The key engineering challenge is that two prompts may share a prefix but have different tokenizations due to the tokenizer state — this edge case must be handled.

#### Interface

```
KVCacheManager:
  allocate(request_id, num_blocks) → list[block_ids]
  free(request_id)
  evict(policy) → list[freed_blocks]
  lookup_prefix(prompt_hash) → list[block_ids] or None
  store_prefix(prompt_hash, block_ids)
  stats() → {total_blocks, free_blocks, fragmentation_pct}
```


### 3.2 Continuous Batching Scheduler

#### Why static batching fails

Static batching groups requests into fixed-size batches and processes each batch to completion. Within a batch, all requests generate tokens in parallel, but the batch latency equals the *longest* request in the batch:

```
Batch [A:512 tokens, B:32 tokens, C:128 tokens, D:64 tokens]
                     ↓
  The batch runs for 512 steps.
  B finishes at step 32, but its slot is wasted for 480 more steps.
  C finishes at step 128, waste 384 steps.
  D finishes at step 64, waste 448 steps.
  A uses every step.
```

Total GPU utilization: `(512 + 32 + 128 + 64) / (512 × 4) = 36%`. The other 64% of GPU time is wasted on padding tokens.

#### Iteration-level scheduling

At each decoding step, the scheduler re-evaluates the batch composition. Finished requests leave; new requests enter. The batch size stays roughly constant, and the GPU is almost never processing padding.

```
Step 1:  [A B C D]    4 active, 0 idle slots
Step 32: [A C D E]    B finished, E enters (from pending queue)
Step 64: [A C E F]    D finished, F enters
Step 128:[A G H I]    C finished, 3 new requests enter
...
Step 512:[A]          A is the last one standing
```

The key invariant: **the scheduler tries to keep `num_active == max_batch_size` at all times**. When a request finishes, the memory manager frees its KV blocks, and the scheduler immediately admits the next request from the queue — without waiting for the current batch to finish.

#### Preemption

What if a high-priority request arrives and the batch is full? The scheduler can preempt a running request: its generated tokens and KV cache are saved, and it rejoins the pending queue. Preemption is expensive (it wastes the work already done), so it should be rare. A simpler alternative: reserve one batch slot for express requests, or use a separate priority queue.

#### Scheduling Policy

| Policy | Strategy | Best For |
|--------|----------|----------|
| FCFS | First-come, first-served | Simple, fair |
| Shortest-first | Admit shortest requests first | Minimizing average latency |
| Priority | Weighted by arrival time × estimated length | Mixed workloads |

For a typical chatbot serving endpoint, FCFS with one reserved slot for system-health checks is sufficient. More complex policies add overhead without measurable user-facing benefit for this workload.

#### Interface

```
Scheduler:
  add_request(Request) → request_id
  step() → Batch (set of active request_ids for this step)
  preempt(request_id) → saved_state
  stats() → {queue_depth, active_count, total_steps, avg_batch_size}
```


### 3.3 Quantization Integration

#### Why quantization in the serving system

Quantization is not a one-time decision made at model export time. The optimal quantization strategy depends on:

1. **The GPU**: A 24GB card allows FP16 for 7B but needs INT4 for 13B. An 80GB A100 can run 70B in FP8.
2. **The workload**: A batch-heavy server benefits more from quantization than a single-request debug instance.
3. **The latency target**: Interactive applications need <500ms TTFT; batch inference can tolerate more.
4. **The quality bar**: Customer-facing chat vs. internal data processing have different accuracy requirements.

#### Supported methods and their tradeoffs

| Method | Memory Reduction | Speed Change | Accuracy Impact | Best Use Case |
|--------|-----------------|--------------|----------------|---------------|
| FP16 | Baseline | Baseline | Baseline | Development, debugging |
| FP8 (E4M3) | 50% | 1.3× faster | <0.1% | GPU with native FP8 support (H100+) |
| INT8 | 39% | 1.1× faster | <0.5% | General purpose, good accuracy |
| AWQ INT4 | 64% | 1.4× faster | ~0.5-1% | Production on consumer GPUs |
| GPTQ INT4 | 61% | 1.3× faster | ~0.7-1.2% | When AWQ not available |

**Why AWQ > GPTQ for serving**: AWQ uses activation-aware scaling — it identifies which 1% of weights matter most for model output and preserves them at higher precision. GPTQ uses layer-wise Hessian-based compensation which has better theoretical properties but slightly lower accuracy in practice. For serving, AWQ's simpler dequantization kernels give better throughput.

#### Integration with the serving pipeline

Quantization is a model-load-time decision, not a runtime decision. When the server starts:

```
1. Read ENGINE_TYPE from env
2. Read QUANTIZATION from env
3. Model loader:
   - FP16: standard load
   - AWQ/GPTQ: load quantized weights, attach dequant stubs
4. Memory Manager: adjust block size based on quantization
   (INT4 models can use larger blocks since memory pressure is lower)
5. Return configured engine
```

The key insight: **the KV Cache Manager should be aware of the quantization level**. With INT4 weights, the model occupies less VRAM, leaving more room for KV cache. The scheduler can scale `max_batch_size` accordingly.


### 3.4 Speculative Decoding

#### The core idea

Autoregressive decoding generates one token per model forward pass. A 7B model at 50ms/token generates 20 tokens/second — fine for chat, but limiting for batch processing.

Speculative decoding exploits the asymmetry between model sizes:
- A 0.5B draft model generates γ tokens auto-regressively in ~8ms/token
- A 7B target model verifies all γ draft tokens in ONE forward pass in ~50ms
- If most draft tokens are accepted, effective throughput approaches `(γ / (γ × 8 + 50))` tokens/ms

#### The rejection sampling algorithm

```
1. Draft model generates γ candidate tokens: [t₁, t₂, ..., t_γ]
2. Target model computes log-probs for ALL γ positions in one pass:
   p_target(t₁), p_target(t₂), ..., p_target(t_γ)
3. For each position i:
   - If p_target(t_i) > p_draft(t_i): ACCEPT (target is more confident)
   - Else: ACCEPT with probability p_target(t_i) / p_draft(t_i)
   - On first rejection: STOP, discard remainder
4. All accepted tokens are committed; one corrected token is generated
5. Loop to step 1 with updated prompt
```

#### When speculative decoding helps (and when it doesn't)

| Condition | Effect on speedup |
|-----------|------------------|
| Draft and target share vocabulary | Higher acceptance rate (same tokenizer) |
| Draft model is ~10-100× smaller | Good cost-quality balance |
| Low temperature (greedy/sampling) | Higher acceptance (less variance) |
| Very long prompts | Draft overhead amortized over many tokens |
| Batch processing | Target forward pass already batched; draft adds marginal cost |

**When NOT to use**: interactive chat with temperature > 0.7. High temperature means low acceptance rate, which negates the benefit.

#### Engineering considerations

The draft model must be loaded alongside the target model, consuming additional VRAM. For a 0.5B draft + 7B target, the draft adds ~4GB (FP16). On a 24GB GPU, this might crowd out KV cache capacity. The tradeoff: speculative decoding helps when you are throughput-bound, not memory-bound.


### 3.5 Context Compression for Chat Applications

#### Why this is part of the serving problem

Most inference optimization focuses on the model side: faster kernels, better batching, lower-precision compute. But there's another lever: **reduce the amount of computation in the first place**.

For chat applications, conversations grow linearly over time. The 50th message in a session generates the same per-step cost as the 1st (KV cache is already computed), BUT the prompt encoding cost grows with message count. Moreover, since most models have context limits (4K–32K tokens), long conversations will eventually exceed the limit.

Context compression sits between the application and the model. It's a pipeline that processes the conversation history before it reaches the model, deciding what to keep and what to discard.

#### The compression decision tree

```
Incoming message + history
        │
        ▼
  Token budget check
        │
        ├── Under budget → Full history (no compression)
        │
        ├── Over budget → Hierarchical compression:
        │     ├── Level 0: Last 5 turns, verbatim
        │     ├── Level 1: Turns 6-20, truncated to 100 chars
        │     ├── Level 2: Turns 21-50, truncated to 50 chars
        │     └── Level 3+: Replaced with [compressed] marker
        │
        └── Severely over → Topic analysis:
              ├── LLM analyzes conversation for topics
              ├── User selects which topic to focus on
              └── Context rebuilt from selected topic history only
```

**Why this works**: The information density of conversation turns decreases with age. The last 5 turns determine the next response ~90% of the time. Beyond that, preserving the *gist* (via compression) is more valuable than preserving the exact wording (which the model doesn't need for a coherent response).

#### Integration with the serving pipeline

Context compression is a pre-processing step. It happens before the prompt reaches the model. This means:
- It can run on CPU (the Go chat service handles it)
- It doesn't add to the GPU's workload
- It's compatible with all optimization techniques below it (batching, quantization, etc.)


## 4. Implementation Plan

### Phase 1: Benchmark Infrastructure (foundation for all measurements)

**Goal**: Every optimization has a before/after number.

```
inference-engine/benchmark/
├── latency_bench.py         # TTFT, TPOT vs. batch size, sequence length
├── throughput_bench.py      # tokens/sec vs. concurrent requests
├── memory_bench.py          # VRAM vs. model size, quantization, KV cache
└── quality_bench.py         # MMLU/GSM8K accuracy vs. quantization level
```

Each benchmark:
1. Accepts configurable parameters (model, batch size, seq length, quantization)
2. Runs multiple trials and reports mean ± std
3. Outputs both human-readable tables and machine-readable JSON
4. Can run in CI (with simulated GPU) for regression testing

### Phase 2: KV Cache Manager Implementation

**File**: `inference-engine/memory-manager/kv_cache_manager.py`

A proper implementation with:
- Block-based allocation pool
- Reference-counted prefix cache
- Configurable eviction policy (LRU, Window, Attention-weighted)
- Integration with the inference engine so that `generate()` calls allocate through the manager

### Phase 3: Continuous Batching Integration

**File**: `inference-engine/scheduler/continuous_batching.py` (already started)

Enhance existing scheduler to:
- Actually integrate with the KV Cache Manager (allocate/free blocks per step)
- Support preemption of running requests
- Measure and log per-step batch composition for analysis

### Phase 4: Quantization Analysis Pipeline

**File**: `inference-engine/benchmark/quantization_bench.py` (already started)

Extend to:
- Run actual model inference (not just reference numbers)
- Compare all supported quantization methods on the same hardware
- Produce memory-accuracy-speed tradeoff matrix


## 5. Non-Goals

Things deliberately excluded from this design:

1. **Custom CUDA kernels** — This project layers on top of existing frameworks (HF, vLLM). Kernel optimization belongs in vLLM itself, not here.
2. **Multi-GPU serving (TP/PP)** — Distributed inference is a separate complexity class. Single-GPU optimization must be solid first.
3. **Online learning / continuous adaptation** — RLHF from live traffic adds operational complexity that few deployments need.
4. **Automatic model selection** — Choosing the right model size for a query is a separate research problem.

---

## Appendix A: Key References

- **Orca (SOSP 2022)**: Iteration-level scheduling for LLM serving. Yu et al.
- **vLLM (SOSP 2023)**: PagedAttention for efficient KV cache management. Kwon et al.
- **H2O (NeurIPS 2023)**: Heavy-hitter oracle for KV cache eviction. Zhang et al.
- **StreamingLLM (ICLR 2024)**: Attention sink phenomenon and sink-aware eviction. Xiao et al.
- **AWQ (MLSys 2024)**: Activation-aware weight quantization. Lin et al.
- **DPO (NeurIPS 2023)**: Direct preference optimization as RLHF alternative. Rafailov et al.
