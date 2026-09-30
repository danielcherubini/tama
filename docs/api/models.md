# Models API

Manage model configurations. Each model maps a HuggingFace repo (`repo_id`) to a backend and runtime settings.

## GET /tama/v1/models

List all model configs plus available backends and sampling templates.

**Response:**

```json
{
  "models": [
    {
      "id": 1,
      "repo_id": "bartowski/Llama-3.1-8B-Instruct-GGUF",
      "backend": "llama_cpp",
      "gpu_variant": "cuda",
      "gpu_device": null,
      "model": null,
      "quant": "Q4_K_M",
      "mmproj": null,
      "mtp_model": null,
      "args": [],
      "sampling": null,
      "enabled": true,
      "context_length": null,
      "num_parallel": null,
      "port": null,
      "api_name": null,
      "display_name": null,
      "kv_unified": true,
      "gpu_layers": null,
      "cache_type_k": null,
      "cache_type_v": null,
      "hf_context_length": null,
      "hf_architecture_type": null,
      "hf_base_model": null,
      "quants": {
        "Q4_K_M": {
          "file": "llama-3.1-8b-instruct-q4_k_m.gguf",
          "kind": "Q4_K_M",
          "size_bytes": 4500000000,
          "context_length": null,
          "lfs_oid": "sha256:abc...",
          "db_size_bytes": 4500000000,
          "last_verified_at": "2025-01-01T00:00:00Z",
          "verified_ok": true,
          "verify_error": null
        }
      },
      "modalities": null,
      "reasoningLevels": null,
      "spec_decoding": {},
      "vllm": {},
      "repo_commit_sha": null,
      "repo_pulled_at": null,
      "capabilities": {
        "supports_mtp": true,
        "has_mtp_draft_file": false,
        "has_mmproj": false
      }
    }
  ],
  "backends": [
    {
      "name": "llama_cpp",
      "type": "LlamaCpp",
      "path": "/path/to/llama_cpp/..."
    }
  ],
  "sampling_templates": {}
}
```

## GET /tama/v1/models/:id

Get a single model config.

**Path params:**
- `id` — Integer ID or config_key (double-dash format, e.g. `bartowski--llama-3.1-8b-instruct-gguf`)

**Response:** Same shape as a single entry from the list endpoint, plus a `"backends"` array with available backend options.

`reasoningLevels` (array or `null`) is the raw stored value only — the client model-info endpoints (`/v1/opencode/models`, `/v1/models`) additionally emit the derived `supportsReasoningEffort` (true when non-empty levels are set) and the opencode-canonical `reasoning_options` (`off`→`none`).

**Errors:** `404 Not Found`

## POST /tama/v1/models

Create a new model config.

**Request body:**

```json
{
  "repo_id": "bartowski/Llama-3.1-8B-Instruct-GGUF",
  "backend": "llama_cpp",
  "gpu_variant": "cuda",
  "gpu_device": null,
  "model": null,
  "quant": null,
  "mmproj": null,
  "mtp_model": null,
  "args": [],
  "sampling": null,
  "enabled": true,
  "context_length": null,
  "num_parallel": null,
  "port": null,
  "api_name": null,
  "display_name": null,
  "gpu_layers": null,
  "quants": {},
  "modalities": null,
  "reasoningLevels": null,
  "kv_unified": true,
  "cache_type_k": null,
  "cache_type_v": null,
  "spec_decoding": null,
  "vllm": null,
  "metadata": null,
  "capabilities": {
    "supports_mtp": true,
    "has_mtp_draft_file": false,
    "has_mmproj": false
  }
}
```

**Field reference:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_id` | string | **Yes** | HuggingFace repo `owner/repo` (max 256 chars, alphanumeric + `.` `_` `-` `/`) |
| `backend` | string | **Yes** | Backend name: `"llama_cpp"`, `"ik_llama"`, etc. |
| `gpu_variant` | string | No | `"cpu"`, `"cuda"`, `"vulkan"`, `"rocm"`, `"metal"` |
| `gpu_device` | string | No | Specific GPU device identifier |
| `model` | string | No | Internal model name override |
| `quant` | string | No | Default quant key (e.g. `"Q4_K_M"`) |
| `mmproj` | string | No | Multi-modal projector quant key |
| `mtp_model` | string | No | Multi-token prediction model quant key |
| `args` | string[] | No | CLI arguments passed to the backend |
| `sampling` | object | No | Sampling parameters |
| `enabled` | bool | No | Whether model is active (default `true`) |
| `context_length` | int | No | Override context length |
| `num_parallel` | int | No | Number of parallel requests |
| `port` | int | No | Custom backend port |
| `api_name` | string | No | Custom name for OpenAI-compatible routes |
| `display_name` | string | No | Human-readable display name |
| `gpu_layers` | int | No | Number of layers on GPU |
| `quants` | object | No | Map of quant key → `QuantEntry` |
| `modalities` | object | No | e.g. `{"text": true, "image": true}` |
| `reasoning_levels` | string[] \| null | No | Reasoning effort levels the model accepts. Valid values: `off, minimal, low, medium, high, xhigh, max` (also accepted as camelCase `reasoningLevels`). Stored as given (trim/lowercase/dedupe applied). When non-empty, the model advertises `supportsReasoningEffort: true` on the client model-info endpoints. PUT/PATCH: `null`/absent preserves the existing value, `[]` clears |
| `kv_unified` | bool | No | Use unified KV cache (default `true`) |
| `cache_type_k` | string | No | KV cache type for keys (e.g. `"f8"`, `"q4"`) |
| `cache_type_v` | string | No | KV cache type for values |
| `spec_decoding` | object | No | Speculative decoding config |
| `vllm` | object | No | vLLM-specific launch settings for transformers-format models |
| `metadata` | object | No | `HfModelMetadata` to pre-populate HF fields |

**Response (201 Created):**

```json
{ "ok": true, "id": 1 }
```

**Errors:**
- `409 Conflict` — `repo_id` already exists
- `422 Unprocessable Entity` — Validation failure (e.g. a `reasoningLevels` entry outside `off, minimal, low, medium, high, xhigh, max`; error type `ValidationError`, message names the offending values and the valid set)

## PUT /tama/v1/models/:id

Update an existing model. Partial update — only provided fields change.

**Request body:** Any subset of the `POST /tama/v1/models` body (minus `repo_id` and `metadata`).

**Response (200 OK):**

```json
{ "ok": true, "id": 1 }
```

**Errors:** `404 Not Found`, `422 Unprocessable Entity`

## PATCH /tama/v1/models/:id

Update an existing model. Surgical partial update — only provided fields change, all others preserved.

**Path params:**
- `id` — Integer ID or config_key (double-dash format, e.g. `bartowski--llama-3.1-8b-instruct-gguf`)

**Request body:** `ModelPatchBody` — all fields optional.

```json
{
  "backend": "llama_cpp",
  "args": null,
  "enabled": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `backend` | string \| null | Backend name (optional — unlike PUT where it was required) |
| `args` | string[] | null | CLI arguments — `null` preserves current value, `[]` clears |
| All other fields | Various | Same as `POST /tama/v1/models` body — all optional, `null` = preserve |

**Response (200 OK):**

```json
{ "ok": true, "id": 1 }
```

**Errors:**
- `404 Not Found` — Model does not exist
- `422 Unprocessable Entity` — Validation failure

## POST /tama/v1/models/:id/rename

Rename a model (change its `repo_id`). The integer `id` is preserved.

**Request body:**

```json
{ "new_repo_id": "new-owner/new-repo-name" }
```

**Response (200 OK):**

```json
{ "ok": true, "id": 1 }
```

**Errors:**
- `404 Not Found` — Model does not exist
- `409 Conflict` — Target `repo_id` already exists
- `422 Unprocessable Entity` — Invalid `new_repo_id` format

## DELETE /tama/v1/models/:id

Delete a model config and all associated files from disk. Removes the model directory, model card, and database records.

**Response (200 OK):**

```json
{ "ok": true }
```

**Errors:** `404 Not Found`

## DELETE /tama/v1/models/:id/quants/:quant_key

Delete a single quant entry from a model and its GGUF file. If the deleted quant was the active `quant` or `mmproj`, those fields are cleared to `null`.

**Response (200 OK):**

```json
{
  "ok": true,
  "id": 1,
  "quant_key": "Q4_K_M",
  "deleted_file": "llama-3.1-8b-instruct-q4_k_m.gguf"
}
```

**Errors:** `404 Not Found` (model or quant does not exist)

## POST /tama/v1/models/:id/refresh

Re-query HuggingFace for the current commit SHA and per-file LFS hashes/sizes, and write them into the local database. Only updates metadata for files already tracked locally.

**Response (200 OK):**

```json
{
  "ok": true,
  "id": 1,
  "repo_id": "bartowski/Llama-3.1-8B-Instruct-GGUF",
  "repo_commit_sha": "abc123...",
  "repo_pulled_at": "2025-01-01T00:00:00Z",
  "files": [
    {
      "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
      "quant": "Q4_K_M",
      "lfs_oid": "sha256:...",
      "size_bytes": 4500000000,
      "downloaded_at": null,
      "last_verified_at": "2025-01-01T00:00:00Z",
      "verified_ok": true,
      "verify_error": null
    }
  ]
}
```

## POST /tama/v1/models/:id/verify

Recompute SHA-256 for every tracked file and compare against stored LFS hashes. CPU-bound and potentially slow for large GGUF files.

**Response (200 OK):**

```json
{
  "ok": true,
  "any_unknown": false,
  "id": 1,
  "repo_id": "bartowski/Llama-3.1-8B-Instruct-GGUF",
  "results": [
    {
      "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
      "ok": true,
      "error": null
    }
  ],
  "files": [ /* same file format as refresh */ ]
}
```

## LiteLLM-compatible discovery

`GET /v1/model/info` and `GET /model/info` (same handler) serve models in
LiteLLM's `model_info_v1` discovery shape, so plugins written against
LiteLLM's discovery API (e.g. `pi-provider-litellm`) work against Tama
unchanged: one `GET /v1/model/info` call returns every model with the full
`model_info` metadata.

**Auth:** `tama_` Bearer API key. Both routes require the `Inference` scope —
`/model/info` is the LiteLLM-compatible twin of `/v1/model/info` (same handler,
same data) and carries the same authorization contract. The auth middleware
still gates both when API keys are enabled.

**Query filter:** `?litellm_model_id=<name>` filters to the single entry whose
`model_name` matches (LiteLLM's per-model lookup). An unknown id yields an
empty `data` array; an absent or empty value returns the full list.

**Response (200 OK):**

```json
{
  "data": [
    {
      "model_name": "qwen3-8b",
      "litellm_params": {
        "model": "qwen3-8b",
        "provider": "llama.cpp"
      },
      "model_info": {
        "id": "qwen3-8b",
        "mode": "chat",
        "litellm_provider": "llama.cpp",
        "max_input_tokens": 262144,
        "max_output_tokens": 32768,
        "max_tokens": 32768,
        "supports_function_calling": true,
        "supports_reasoning": true,
        "supports_vision": true,
        "supports_audio_input": false,
        "supports_pdf_input": false,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "cache_read_input_token_cost": 0.0,
        "cache_creation_input_token_cost": 0.0,
        "reasoning_effort_levels": ["none", "low", "medium", "xhigh"],
        "supports_none_reasoning_effort": true,
        "supports_minimal_reasoning_effort": false,
        "supports_low_reasoning_effort": true,
        "supports_xhigh_reasoning_effort": true,
        "supports_max_reasoning_effort": false
      }
    }
  ]
}
```

One entry per enabled model plus alias entries (same set and inheritance rules
as `/v1/opencode/models`). The example above is a Qwen3.8-shaped entry with
`reasoning_levels: ["off", "low", "medium", "xhigh"]` stored — note the
`off` → `none` conversion (ADR-0009) in `reasoning_effort_levels` and the
matching `supports_none_reasoning_effort: true`.

**Field mapping:**

| LiteLLM field | Tama source |
|---------------|-------------|
| `model_name` | `api_name` / `model` (or alias name) |
| `litellm_params.model` / `model_info.id` | Same |
| `litellm_params.provider` / `model_info.litellm_provider` | Backend name |
| `max_input_tokens` | Context-length resolution chain: config override → vLLM `max_model_len` → live backend → HF metadata → model TOML |
| `max_output_tokens` | 1/8 of context, clamped 16K–32K (same heuristic as the opencode `limit.output`) |
| `max_tokens` | `max_output_tokens` else `max_input_tokens` (LiteLLM's legacy field) |
| `supports_function_calling` | Backend `/props` |
| `supports_reasoning` | `/props` reasoning OR non-empty `reasoning_levels` (ADR-0008) |
| `supports_vision` / `supports_audio_input` / `supports_pdf_input` | `modalities.input` (`image` / `audio` / `pdf`) |
| Cost fields (all four) | `0.0` (local inference — no per-token pricing) |
| Effort fields (`reasoning_effort_levels`, `supports_*_reasoning_effort`) | `reasoning_levels` with `off` → `none` (ADR-0009); **absent** when no levels are configured (LiteLLM treats absent as "unknown — advertise nothing") |

Note the `off`/`none` duality (ADR-0009): Tama stores levels in the pi
vocabulary (`off`) and converts to the wire vocabulary (`none`) at
serialization only. `medium` and `high` have no per-level flag in LiteLLM's
schema — they appear only inside `reasoning_effort_levels`.

**Known limitation:** `GET /health` is a liveness probe
(`{ "status": "ok", "service": "tama-proxy" }`), **not** LiteLLM's
`{ "healthy_endpoints": [...] }` — plugins whose discovery falls back to
`/health` (only reached when `/v1/model/info` is unreachable or empty) get
zero models from Tama. The `/v1/model/info` primary path is fully supported.

## Model Lifecycle (Load / Unload / Cancel)

Local (self-hosted) model processes are owned by a **tamad** (ADR-0010): the
proxy never spawns or kills a backend process itself. These endpoints resolve
the model's owning provider, build the fully resolved launch spec from the
central database (installation args/env, binary path, model file path, GPU
isolation env, health URL), and dispatch `LoadModel` / `UnloadModel` RPCs to
the provider's tamad.

### POST /tama/v1/models/:id/load

Load a model on its provider's tamad. The model is marked **desired** in the
proxy database; the reconciler (see below) keeps it alive.

- If the model's provider has no tamad assigned, the request fails with a
  clear error (`Provider "<name>" has no tamad assigned`).
- The load fails fast when the target tamad reports all of its GPUs ≥ 95%
  VRAM used and the model needs a GPU.
- LRU eviction (`max_loaded_models`) operates on the proxy's mirror of the
  tamads' process tables before the load.

**Response (200 OK):** `{ "id": "<model>", "loaded": true }`

**Errors:** `500` with `LoadModelError` (provider/tamad resolution, RPC, or
health-check failure on the tamad).

### POST /tama/v1/models/:id/unload

Clear the model's **desired** state and issue `UnloadModel` to the
provider's tamad. Unloading a model that is not loaded on the tamad is a
no-op (idempotent).

**Response (200 OK):** `{ "id": "<model>", "loaded": false }`

### POST /tama/v1/models/:id/cancel

Cancel a load: clears the **desired** state and issues a best-effort
`UnloadModel` to the tamad. Cancelling a load that is still in flight (the
tamad is still health-polling) is best-effort — the reconciler unloads the
model on its next tick (~1s) once it appears in the tamad's process
snapshot.

**Response (200 OK):** `{ "id": "<model>", "loaded": false }`

**Errors:** `404` with `ModelNotLoadingError` when the model is neither
desired nor loaded anywhere.

## Desired vs Actual Model State

- **Desired state** is the proxy's intent, stored in the central database
  (`desired_models`): which models should be loaded on which tamad. It
  survives proxy *and* tamad restarts.
- **Actual state** is each tamad's in-memory process table, streamed to the
  proxy in the per-second stats snapshot (`SystemStats.processes`).
- A proxy-side **reconciler loop** (1s tick) converges actual to desired:
  - desired but missing or dead → re-issue `LoadModel` (bounded by
    `max_restarts` within a 5-minute window per model),
  - running but no longer desired → `UnloadModel`.
- Consequences: a killed backend process is respawned within ~2s; after a
  tamad restart its desired models auto-load; after a proxy restart the
  reconciler's first tick converges from the persisted desired set.
- The proxy keeps a short-lived local mirror of the tamads' process tables
  so the forward path and `GET /tama/v1/models` report live endpoints.
