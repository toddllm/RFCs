# Async Collective Overlap For Spyre Tensor Parallel Inference

**Authors:**

* @toddllm

## **Summary**

Tensor-parallel (TP) inference on Spyre needs to overlap collective
communication with local compute. Today the Spyre `torch.distributed`
backend rejects `async_op=True` for collectives, so callers cannot
issue a collective and continue with independent compute. This RFC
proposes:

1. A schedule-scoped completion contract in the lower communication
   runtime so a `WorkSchedule`'s `wait()` and `query()` reflect that
   schedule's own work, not the entire shared device stream.
2. A real `Work` handle for `dist.broadcast(..., async_op=True)` in the
   Spyre `torch.distributed` backend.
3. A TP=2 fallback in the Spyre Inference communicator that exposes
   `SpyreCommunicator.all_reduce(input_, async_op=True)` and pipelines
   multiple in-flight reductions through two symmetric broadcasts.

The scope is the async collective substrate. The RFC defines the
runtime, c10d, and Spyre Inference pieces required for a caller to
start a collective, run independent work, and wait on that collective
later. Model-level scheduling policy and vLLM graph rewrites can build
on this substrate in later work.

The result is the standard PyTorch user shape:

```python
work = dist.broadcast(tensor, src=0, group=group, async_op=True)
do_independent_work()
work.wait()
```

```python
work = communicator.all_reduce(tensor, async_op=True)
do_independent_work()
tensor = work.wait()
```

In one sentence: make Spyre collectives behave like normal PyTorch
async collectives, then use that shape to create a first TP=2
overlap path for inference.

## **Motivation**

Tensor parallelism splits one model layer across multiple devices.
Each rank computes part of the result; some layers then need the ranks
to combine partial results.

The combination step is a collective. For inference it is on the
critical path because every token runs through the same layer stack.
If each layer waits for communication before starting the next useful
piece of compute, the per-layer communication cost accumulates into
per-token latency.

The useful mental model is:

```text
one token
  |
  v
+---------------------+     +---------------------+
| transformer layer 0 | --> | transformer layer 1 | --> ...
+---------------------+     +---------------------+
      |                           |
      v                           v
  local matmul                local matmul
      |                           |
      v                           v
  TP collective               TP collective
```

The local matmul work and the collective work use different parts of
the stack. When the runtime can keep the collective in flight and the
caller has independent compute available, the communication latency can
be partially hidden.

For TP=2 the contributing operation is a sum-allreduce:

```text
rank 0 partial tensor   \
                         + ----> summed tensor on both ranks
rank 1 partial tensor   /
```

A naive synchronous schedule pays the full collective cost on every
layer:

```text
time -->

[ compute N ][ allreduce N ][ compute N+1 ][ allreduce N+1 ]
```

The desired async schedule starts the collective, runs independent
work while it is in flight, and waits only when the reduced value is
needed:

```text
time -->

[ start allreduce N         ][ wait(N) ][ use reduced tensor N ]
                [ compute N+1 (independent work) ]
```

The first implementation milestone is collective pipelining. A caller
issues several collectives without waiting for each one immediately,
then waits later:

```text
time -->

sync:
[ allreduce 0 ][ allreduce 1 ][ allreduce 2 ][ allreduce 3 ]

async batch:
[ start 0 ][ start 1 ][ start 2 ][ start 3 ][ wait all ]
```

This milestone proves that the collective implementation can keep more
than one operation in flight without each `wait()` behaving like a
global barrier. A separate compiled-compute gate then proves that the
same substrate can overlap communication with an unrelated compute
kernel.

This only works when `wait()` is scoped to the work that was started.
If `wait()` drains the whole shared runtime stream, unrelated compute
and unrelated collectives are pulled across the synchronization
boundary and the overlap window collapses.

The required completion model is per-schedule, not per-stream:

```text
WorkSchedule A:
  op A1 -> op A2 -> completion fence A

WorkSchedule B:
  op B1 -> op B2 -> completion fence B

wait(A)  waits for fence A
wait(B)  waits for fence B
query(A) reports fence A state
query(B) reports fence B state
```

By contrast, a global drain has the wrong shape:

```text
Shared stream:
  A1 -> A2 -> fence A -> B1 -> B2 -> fence B

global_wait(A):
  waits for all currently visible stream work
  can accidentally include B

schedule_wait(A):
  waits only for fence A
```

The distinction matters because PyTorch `Work` handles are logical
objects. A caller expects `work_a.wait()` to wait for operation A, not
for every other operation that happens to share the same device stream.

For TP-parallel inference this matters at three levels:

1. **Inference layer.** vLLM-style tensor-parallel layers issue an
   allreduce per `RowParallelLinear` and per `down_proj`. A 40-layer
   model performs roughly `1 + 2L` allreduces per forward (one for the
   embedding-output sum and two per transformer layer), so even a
   small per-allreduce reduction multiplies. The async shape lets
   independent compute (the next layer's MLP, for example) hide a
   slice of each collective.
2. **Distributed backend.** The PyTorch user contract is
   `dist.broadcast(tensor, src=..., async_op=True)` returning a
   `Work` whose `wait()` is scoped to that broadcast. Today's Spyre
   backend cannot honor that contract.
3. **Communication runtime.** The schedule-scoped completion model is
   what lets multiple in-flight schedules share a device stream
   without observing each other's progress.

## **Proposed Implementation**

### Responsibilities By Layer

The design has three implementation layers. Each layer has a small
responsibility and a clear contract with the layer above it:

| layer | responsibility |
|---|---|
| Communication runtime | Provide `WorkSchedule::start()`, `wait()`, `query()`, and `reset()` with schedule-scoped completion. |
| Torch-Spyre c10d backend | Return a real PyTorch `Work` object for `dist.broadcast(..., async_op=True)` and map `Work::wait()` / `isCompleted()` to the schedule. |
| Spyre Inference communicator | Expose `SpyreCommunicator.all_reduce(..., async_op=True)` for TP=2 by composing two broadcasts and deferring the add until `wait()`. |

The design follows the normal PyTorch async shape. That keeps
application code simple and gives later vLLM integration work the same
control surface it already expects from other distributed backends.

### Layered Call Path

The intended runtime call path:

```text
+-----------------------------------------------------------+
| Spyre Inference (vLLM tensor-parallel layer)              |
|   SpyreCommunicator.all_reduce(input_, async_op=True)     |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
| torch.distributed                                         |
|   dist.broadcast(tensor, src=..., async_op=True)          |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
| Torch-Spyre c10d backend                                  |
|   SpyreCCLBackend::broadcast(...)  -> Work handle         |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
| Communication runtime                                     |
|   WorkSchedule::start() / wait() / query() / reset()      |
|   Schedule-scoped completion fence                        |
+-----------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------+
| Spyre device stream / hardware                            |
+-----------------------------------------------------------+
```

### Schedule-Scoped Completion (Communication Runtime)

The communication runtime exposes a `WorkSchedule` object that
collects the operations for one logical collective. Operations are
launched onto a shared per-stream device queue.

For the async user shape to work, the schedule contract is:

* `start()` submits the schedule's operations.
* `wait()` blocks until **this schedule's** operations are complete.
* `query()` returns true only when this schedule is complete.
* `reset()` returns the schedule to a clean re-launchable state.
* Empty schedules are safe (`wait()` and `query()` return without
  side effects).
* Repeated `wait()` is safe.
* Destroying a `Work` handle without waiting does not invalidate
  in-flight runtime callbacks.

The implementation appends a per-schedule completion fence as the last
operation submitted in `start()`. The fence lifetime is independent of
the schedule object: callers that discard a `Work` without waiting do
not race with in-flight callbacks.

The contract this RFC proposes is API-level. The exact fence
mechanism is a runtime implementation detail; what matters at this
layer is that `wait(A)` does not wait for `B` and vice versa.

### `dist.broadcast(async_op=True)` (Torch-Spyre)

The Spyre c10d backend exposes a `broadcast(...)` entry point that
already creates a `WorkSchedule` for the underlying broadcast. The
required behavior change is small:

* `broadcast(async_op=False)`: keeps the historical synchronous
  shape: submit the schedule, block on `wait()`, return a completed
  `Work`.
* `broadcast(async_op=True)`: submit the schedule and return the
  `Work` immediately, **without** the inline `wait()`.
* `Work::wait()`: drives the underlying schedule's `wait()` when one
  is attached.
* `Work::isCompleted()`: drives the underlying schedule's `query()`
  when one is attached.

Code shape:

```cpp
c10::intrusive_ptr<Work> SpyreCCLBackend::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  auto work = c10::make_intrusive<SpyreCCLWork>(OpType::BROADCAST);
  // ... prepare descriptors / get the schedule ...
  work->schedule_ = group_context_->broadcast(tensor, opts.rootRank);
  work->schedule_->start();
  if (!opts.asyncOp) {
    work->schedule_->wait();
  }
  return work;
}

bool SpyreCCLWork::wait(std::chrono::milliseconds /*timeout*/) {
  if (schedule_) schedule_->wait();
  return true;
}

bool SpyreCCLWork::isCompleted() {
  if (schedule_) return schedule_->query();
  return true;
}
```

Synchronous callers see no behavior change. Async callers receive a
`Work` whose `wait()` is scoped to the broadcast they issued.

### TP=2 `all_reduce(async_op=True)` (Spyre Inference)

While the lower communication runtime does not yet implement a native
allreduce on Spyre, the inference communicator can compose an
allreduce from broadcasts. The current sync fallback uses
`recv -> add -> broadcast`, which has a data dependency between
`recv`'s output and the following broadcast's input on the root rank,
so multiple in-flight allreduces serialize through that root.

The proposed fallback uses two symmetric broadcasts and a deferred
add:

```text
rank 0 issues:                rank 1 issues:
  broadcast(A, src=0)           broadcast(peer, src=0)
  broadcast(peer, src=1)        broadcast(B, src=1)

After both complete:
  rank 0:  A.add_(peer)         rank 1:  B.add_(peer)

Result on both ranks: A + B
```

The important ordering point is that both ranks issue the same two
broadcasts in the same order. Only the tensor each rank contributes is
different:

```text
operation order:

  1. broadcast from rank 0
  2. broadcast from rank 1
  3. local add after both broadcasts complete

rank 0 contributes A to op 1 and receives B from op 2
rank 1 receives A from op 1 and contributes B to op 2
```

The two broadcasts are independent (no value-level data dependency
between them), so multiple `all_reduce(async_op=True)` calls in flight
pipeline through the schedule-scoped completion fence above.

The user shape:

```python
work = communicator.all_reduce(tensor, async_op=True)
do_independent_work()
tensor = work.wait()
```

The work handle owns the ordering:

```python
class _SpyreAllReduceWork:
    def __init__(self, *, input_, peer, w_self, w_peer):
        self._input = input_
        self._peer = peer
        self._w_self = w_self
        self._w_peer = w_peer
        self._done = False

    def wait(self):
        if self._done:
            return self._input
        self._w_self.wait()
        self._w_peer.wait()
        self._input.add_(self._peer)
        self._done = True
        return self._input
```

Synchronous callers (`async_op=False`, the default) get back the
in-place reduced tensor and observe no API change. Both ranks issue
broadcasts in matching `src` order so the underlying message matcher
pairs the two halves correctly.

This fallback covers TP=2. Larger world sizes are an explicit
out-of-scope item until either a native allreduce lands or a more
general pattern is designed.

### Current Prototype Status

The current public candidate work has three useful proof points:

* The TP layer correctness probes pass for the layer surfaces that use
  tensor-parallel communication.
* A collective-batch probe shows wall-clock savings when multiple
  `all_reduce(async_op=True)` calls are issued before waiting.
* The candidate branches can express the intended user shape at the
  Spyre Inference and Torch-Spyre layers.

Those proof points are intentionally narrow. They show that the async
collective substrate is plausible and that the TP=2 fallback can be
validated independently. The remaining gates are:

* a current-source communication-runtime build that includes the
  schedule-scoped completion contract;
* a compiled compute plus async collective probe that demonstrates
  compute/communication overlap on top of the collective-pipelining
  substrate;
* a full model-path integration that shows where the async calls are
  introduced in the inference layer.

## **Metrics**

Validation should produce four distinct kinds of evidence. Each
addresses a different question and uses a different probe shape; they
are not interchangeable.

### Correctness evidence

A TP layer correctness suite confirms the candidate stack does not
regress synchronous behavior.

| field | value |
|---|---|
| What | Forward output of each TP-aware layer (`RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`, `VocabParallelEmbedding`, the bare `tensor_model_parallel_all_reduce`) compared against a single-rank reference within fp16 tolerance. |
| Why this gate | The synchronous path is the surface stock vLLM exercises today; this catches any regression introduced by the schedule-scoped fence or the two-broadcast `all_reduce`. |
| Opening pass criterion | 5/5 PASS on both baseline and candidate. |

### Pipelining evidence

A collective-batch async probe confirms that multiple in-flight
`all_reduce(async_op=True)` calls actually pipeline through the
schedule-scoped fence.

| field | value |
|---|---|
| What | Issue `N` `all_reduce(async_op=True)` calls upfront, do a fixed compute window, then wait at the end. Compare wall-clock against the same `N` calls issued synchronously back-to-back through the same communicator on the same hardware. |
| Why this gate | Without schedule-scoped completion, async batches fall back to the synchronous wall-clock; the saving is the direct test that the fence is not a global drain. |
| Opening pass criterion | At least **12%** wall-clock reduction on a 24-layer pattern with `H=4096`, `iters_per_layer=8`. |

### Overlap evidence

A compiled compute + async allreduce probe confirms that an async
allreduce can run alongside a separately submitted compute kernel.

| field | value |
|---|---|
| What | Run a `torch.compile`-d compute kernel concurrently with an async allreduce. Report both the saved wall-clock vs serial `compute -> allreduce`, and the realized fraction of the theoretical overlap window (`max(compute, allreduce) / serial`). |
| Why this gate | Pipelining (above) proves multiple collectives can overlap each other. This gate proves a collective can overlap an unrelated compiled compute kernel. It catches regressions where the schedule fence accidentally drains submitted compute. |
| Opening pass criterion | At least **50% of theoretical overlap realized** at the smallest tensor sizes the compiler will lower. |

### Model-path evidence

A representative TP=2 model forward reports cumulative `all_reduce`
call count and zero comms-layer failures.

| field | value |
|---|---|
| What | Run a 40-layer TP=2 forward (Granite-style architecture is a fair shape) and count cumulative `all_reduce` calls per token, plus comms-layer error count. |
| Why this gate | Confirms the comms substrate carries the full model path without errors. Provides the denominator for any later per-token saving claim. The text quality of generated tokens is **not** an indicator at this gate; it is governed by separate model-layer numerical behavior, especially fp16 paths in OOT layers. |
| Opening pass criterion | Allreduce count equals `1 + 2L` for an `L`-layer TP=2 forward; comms-layer error count is zero. |

## **Drawbacks**

* **On-wire op pattern changes for sync callers too.** The TP=2
  `all_reduce` candidate replaces the existing
  `recv -> add -> broadcast` pattern with two symmetric broadcasts and
  a deferred add. The reduced value is identical to within fp16
  reduction-order tolerance. Reviewers comparing profiling captures or
  message traces will see the new pattern in synchronous runs as well
  as async ones.
* **Per-schedule fence overhead is paid on every schedule.** The
  completion fence is the mechanism that makes `wait()` schedule-scoped.
  Sync callers that never set `async_op=True` still pay the fence
  cost. The validation suite's correctness gate is the place to bound
  that cost; if a measured overhead exceeds the gate's tolerance, the
  fence implementation needs revisiting before landing.
* **TP > 2 remains explicitly out of scope.** The two-broadcast
  fallback is correct only for TP=2 (each rank broadcasts its own
  partial). Larger world sizes need either a native allreduce or a
  different pattern; that is a separate design.
* **Fence ownership crosses repositories.** The schedule-scoped
  contract is enforced in the lower runtime layer, which lives in a
  separate repository and review surface. Coordinating the runtime
  change with the c10d backend change and the inference change
  requires three reviews to land in compatible shape.

## **Alternatives**

1. **Wait for a native runtime allreduce.** This avoids the TP=2
   fallback entirely. It also blocks all overlap work behind a
   larger native-allreduce design. The TP=2 fallback is intended as
   the smallest incremental shape that lets the upper layers reach
   the async user contract today.
2. **Per-stream wait at the c10d layer.** The c10d backend could
   return a `Work` whose `wait()` calls a global stream synchronize.
   The PyTorch type contract is satisfied. The overlap evidence gate
   in the Metrics section, however, requires that an async allreduce
   can run alongside an unrelated submitted compute kernel; a global
   synchronize pulls that compute across the same boundary, so the
   overlap fraction collapses to zero. The schedule-scoped contract
   is what allows the overlap evidence gate to pass.
3. **Split-broadcast with a cached `WorkScheduleInfo`.** An
   alternative shape would split `broadcast()` into a setup phase
   that returns a reusable `WorkScheduleInfo` and an apply phase
   that binds it to a specific tensor on each call. That avoids
   recreating the schedule descriptor every call. Cached reuse is
   not proven safe under the current runtime, so this RFC's
   candidate uses a plain `broadcast()` per call and relies on the
   schedule fence for pipelining. Cached `WorkScheduleInfo` is a
   reasonable follow-up once a safe reuse pattern is established.

## **Prior Art**

* PyTorch's NCCL backend exposes `Work`-handle semantics for
  `dist.broadcast(async_op=True)` and `dist.all_reduce(async_op=True)`
  with stream-aware completion. The user contract proposed here
  matches that shape so application code does not need a Spyre-only
  branch.
* Most modern accelerator runtimes expose a per-event or per-stream
  completion query rather than only a global synchronize. The
  schedule-scoped contract proposed here aligns with that pattern.
* Within Spyre Inference, the existing TP fallback in
  `SpyreCommunicator.all_reduce` already composes an allreduce from
  point-to-point primitives. The two-broadcast variant is a
  refinement of that pattern that removes the inter-call data
  dependency.

## **How we teach this**

Suggested terminology:

* **Schedule-scoped completion.** A `WorkSchedule`'s `wait()` and
  `query()` reflect only that schedule's submitted operations.
* **Async user shape.** `work = op(..., async_op=True)`, do
  independent work, `work.wait()`. This is the shape PyTorch users
  already expect from other distributed backends.
* **Two-broadcast TP=2 allreduce fallback.** Each rank broadcasts its
  partial; both ranks add the peer's partial after both broadcasts
  complete. Composes from the smallest set of primitives currently
  supported.

Documentation impact:

* Spyre Inference: the `SpyreCommunicator.all_reduce` docstring should
  document the new `async_op` parameter, the work-handle shape, and
  the TP>2 raise.
* Torch-Spyre: the c10d backend docs should clarify that
  `dist.broadcast(async_op=True)` is supported and that the returned
  `Work` is schedule-scoped.
* Communication runtime: the `WorkSchedule` docs should pin the
  per-schedule completion contract so future runtime changes do not
  silently revert to a global synchronize.

Existing PyTorch users on other accelerators will not need to learn a
new pattern; the goal is parity with the standard async user shape.

## **Unresolved questions**

* **Native allreduce.** When a runtime-native allreduce lands, the
  TP=2 two-broadcast fallback should be replaced by a direct call.
  The RFC should be revisited to define the deprecation path for the
  fallback.
* **TP > 2.** The fallback is TP=2 only by construction (each rank
  broadcasts its own partial). A general pattern (ring, recursive
  doubling, or a runtime-native allreduce) is needed before TP > 2 can
  be supported through this path.
* **Caller-side overlap inside vLLM.** Current vLLM
  `RowParallelLinear` calls allreduce synchronously. Realizing the
  overlap window on a `LLM.generate()` path needs either a Spyre
  out-of-tree linear-layer override or an upstream vLLM change. That
  is a follow-up RFC, not part of this one.
* **Runtime API pairing.** The communication runtime API has been
  evolving; a stable runtime API version that includes the
  schedule-scoped contract should be pinned for this RFC's landing.
* **Final ownership of the schedule fence.** This RFC describes the
  contract at the API level. The exact owner of the fence
  implementation (runtime layer vs. backend layer) is an
  implementation choice; the runtime layer is preferred because it
  composes naturally with all consumers, but a backend-layer fence
  is also acceptable as long as it satisfies the contract.

## Resolution

Pending review.

### Level of Support

Pending.

### Next Steps

* Land the schedule-scoped completion contract in the communication
  runtime.
* Land the c10d backend `Work`-handle behavior in Torch-Spyre.
* Land the TP=2 `all_reduce(async_op=True)` fallback in Spyre
  Inference.
* Wire the validation probes into a CI lane that runs them against
  the TP=2 device.

#### Tracking issue

To be assigned.
