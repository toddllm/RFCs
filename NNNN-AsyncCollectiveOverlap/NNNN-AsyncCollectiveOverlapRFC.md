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

## **Motivation**

Tensor parallelism splits one model layer across multiple devices.
Each rank computes part of the result; some layers then need the ranks
to combine partial results.

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
  shape — submit the schedule, block on `wait()`, return a completed
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

## **Metrics**

Validation should produce four kinds of evidence:

| metric | what it measures |
|---|---|
| TP layer correctness | Forward output of each TP-aware layer (`RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`, `VocabParallelEmbedding`, the bare `tensor_model_parallel_all_reduce`) matches a single-rank reference within fp16 tolerance. Gates correctness regressions on the synchronous path that all of stock vLLM exercises today. |
| Async batch saved wall-clock | A fixed-pattern probe issues N `all_reduce(async_op=True)` calls upfront and waits at the end, compared to the same N calls issued synchronously back-to-back. Reports percent reduction in wall-clock. |
| Compiled compute + communication overlap | A small probe runs an independent compiled compute kernel concurrently with an async allreduce, then reports both the saved wall-clock and the realized fraction of the theoretical overlap window (`max(compute, comm) / serial`). Catches regressions where the schedule fence accidentally drains unrelated compute. |
| Model-path allreduce count | A representative TP=2 model forward (e.g. a 40-layer Granite-style architecture) reports the cumulative number of `all_reduce` calls per token. Confirms the comms substrate carries the full model path without errors and gives a denominator for any later per-token saving claim. |

A specific quantitative target is appropriate per layer; reasonable
opening targets are:

* TP layer correctness: **5/5 PASS** on baseline and candidate.
* Async batch saved: **>= 12%** vs blocking-serial on a 24-layer
  pattern with `H=4096`, `iters_per_layer=8`. Both runs through the
  same communicator on the same hardware.
* Compiled compute+comm overlap: **>= 50% of theoretical** at the
  smallest tensor sizes the compiler will lower.
* Model-path allreduce count: matches `1 + 2L` for an `L`-layer
  TP=2 forward with no comms-level failures.

## **Drawbacks**

* **Behavior split between sync and async paths.** The TP=2 `all_reduce`
  candidate switches the synchronous path from `recv -> add ->
  broadcast` to two symmetric broadcasts. The reduced value is
  identical, but the on-wire op pattern changes for sync callers too.
  Where exact op patterns are observable (e.g. profiling captures), a
  reviewer should expect the new pattern.
* **Per-WorkSchedule fence cost.** The completion fence adds a small
  per-schedule overhead. The synchronous path observes this cost
  every time. For workloads that never use `async_op=True` this is
  pure overhead.
* **TP=2 only at the inference layer.** Until a native allreduce is
  implemented, TP > 2 must continue to raise. The RFC narrows the
  ask to TP=2 deliberately; any larger pattern is a separate design.
* **Fence is in the lower runtime.** The schedule-scoped contract
  must be implemented in the runtime layer, which is a separate
  repository and review surface.

## **Alternatives**

1. **Wait for a native runtime allreduce.** This avoids the TP=2
   fallback entirely. It also blocks all overlap work behind a
   larger native-allreduce design. The TP=2 fallback is intended as
   the smallest incremental shape that lets the upper layers reach
   the async user contract today.
2. **Per-stream wait at the c10d layer.** The c10d layer could
   continue to drain the runtime stream on every `wait()`, returning
   a real `Work` whose `wait()` simply forwards to a stream
   synchronize. This satisfies the type contract but defeats the
   purpose: any unrelated submitted compute on the same stream is
   pulled across the synchronization boundary, the overlap window
   collapses, and probes that look for overlap show none.
3. **Split-broadcast with a cached `WorkScheduleInfo`.** An earlier
   proof-of-concept attempted to reuse a cached `WorkScheduleInfo`
   and apply it per-call to amortize setup. Caching was not proven
   safe under the current runtime; the candidate path here uses a
   plain `broadcast()` per call and relies on the schedule fence for
   pipelining instead of a cached info. Caching can be revisited as
   a follow-up if a safe reuse pattern is established.

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
