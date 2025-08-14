# Runtime Flow

The procedural finite state machine (FSM) in `amduda_core` coordinates token
processing by reacting to `RuntimeEvent`s emitted by the `aurex-runtime`. The
runtime schedules the next event whenever `Runtime::step` is invoked, allowing
the FSM to progress without relying on a sequence generator.

## States and Transitions

```text
          ┌──────────────┐
          │  FetchToken  │
          └──────┬───────┘
                 │ TokenFetched{cache_hit:true}
                 │                         ┌──────────────┐
                 ├────────────────────────▶│ComputeAttention│
                 │                         └──────┬───────┘
                 │ TokenFetched{cache_hit:false}   │ AttentionComputed
          ┌──────▼───────┐                        │
          │ KVCacheUpdate│◀──────────────────────┘
          └──────┬───────┘ CacheUpdated
                 │
                 ▼
          ┌──────────────┐
          │  OutputToken │
          └──────┬───────┘
                 │ TokenEmitted
                 ▼
          ┌──────────────┐
          │  FetchToken  │
          └──────────────┘

          Rollback ─────▶ FetchToken
          Error    ─────▶ Error
```

* **FetchToken** – Request the next token from the model. A cache hit moves
  directly to **ComputeAttention**, while a miss requires **KVCacheUpdate**.
* **KVCacheUpdate** – Populate the key/value cache before attention.
* **ComputeAttention** – Perform attention using the current token and cache.
* **OutputToken** – Emit the token and transition back to **FetchToken**.
* **Error** – Terminal state entered on unrecoverable failures.
* **Rollback** – Allows reverting to **FetchToken** from any non-error state.

This FSM is driven entirely by runtime events, enabling asynchronous and
streamed token generation across different backends.

