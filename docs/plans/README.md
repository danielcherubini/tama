# Implementation Plans

Plans for features, refactors, and bug fixes in the Tama project.

## Status Legend

| Status | Meaning |
|--------|---------|
| **Backlog** | Plan written and ready to execute |
| ✅ **COMPLETED** | Fully implemented, verified via git history |
| 🔁 **SUPERSEDED** | Replaced by another plan |

## Quick Stats

- **Total Plans**: 102
- **Backlog**: 0
- **Completed**: 100 ✅

> **Note:** The Tama Management API Spec (2026-04-03) was removed as it was a design document, not an implementation plan.

---

## Completed (Recent)

| # | Plan | Status |
|---|------|--------|
| 196 | [LiteLLM-Compatible Model Info](done/plan-196-litellm-model-info.md) | ✅ COMPLETED (squash `3268e5d8`) |
| 195 | [Structured Logging Redesign](done/plan-195-structured-logging.md) (ADR-0013) | ✅ COMPLETED (squash `93985db4`) |
| 194 | [vLLM Spec-Decode Telemetry + Consistent tok/s Formatting](done/plan-194-vllm-spec-telemetry.md) (ADR-0012) | ✅ COMPLETED (squash `930ee31d`) |
| 194 | [Models Page: Live State, Filter Toolbar, Pull-Host UX](done/plan-194-models-live-state-filters-pull-host.md) | ✅ COMPLETED (squash `d59398f4`) |
| 194 | [Detached Health Gate](done/plan-194-detached-health-gate.md) | ✅ COMPLETED (squash `221bbb6e`) |
| 193 | [Tamad is the Source of Truth for Lifecycle](done/plan-193-tamad-lifecycle-truth.md) (ADR-0011) | ✅ COMPLETED (PR #177) |
| 192 | [Gateway Dashboard Refactor & Telemetry](done/plan-192-gateway-dashboard-telemetry.md) | ✅ COMPLETED (squash `28f12ad4`) |
| 191 | [Tamad Host Runtime Split](done/plan-191-tamad-host-runtime.md) | ✅ COMPLETED (squash `05f1694f`) |
| 190 | [SQLite → Postgres Database Migration](done/plan-190-sqlite-to-postgres.md) | ✅ COMPLETED |
| 189 | [Model Reasoning Effort](done/plan-189-model-reasoning-effort.md) | ✅ COMPLETED |
| 188 | [hf CLI Repo Pull (Safetensors Wizard Support)](done/plan-188-hf-cli-repo-pull.md) | ✅ COMPLETED |
| 089 | [VLLM Spec Decoding Settings](done/plan-089-vllm-spec-decoding.md) | ✅ COMPLETED |
| 091 | [VLLM Spec Attention Backend](done/plan-091-vllm-spec-attention-backend.md) | ✅ COMPLETED |
| 090 | [Consolidate Models Page](done/plan-090-consolidate-models-page.md) | ✅ COMPLETED |

---

**All completed plans**: [done/](done/)

---

## Directory Structure

- `docs/plans/` — Backlog plans + this README
- `docs/plans/done/` — Completed plans (archived)
- `docs/plans/backlog.md` — Full backlog with phase descriptions
- `docs/plans/done.md` — All completed plans organized by category
- `docs/plans/execution-order.md` — Dependency-first cascade with phases and coordination flags

## How to Use This Directory

1. **Find a plan** — Browse the Backlog table above or read [backlog.md](backlog.md) for phase descriptions
2. **Check execution order** — See [execution-order.md](execution-order.md) for dependencies and priority
3. **Read the plan** — Understand the goal, architecture, and tasks
4. **Verify implementation** — Follow PR numbers or git references in [done.md](done.md)

## Contributing

When implementing a new feature:

1. Create a new plan file as `docs/plans/plan-NNN-<feature>.md` (NNN is the next sequential number)
2. Follow the template: Goal, Architecture, Tech Stack, Tasks
3. Mark tasks as `[ ]` (not started) or `[x]` (completed)
4. Link to related plans when applicable
5. When complete, move the plan file to `done/` and update [done.md](done.md)

## Related Files

- [`README.md`](../README.md) — Project overview
- [`AGENTS.md`](../AGENTS.md) — Development guide and conventions
- [`docs/openapi/tama-api.yaml`](../openapi/tama-api.yaml) — Machine-readable OpenAPI spec
- [`docs/openapi/openai-compat.yaml`](../openapi/openai-compat.yaml) — OpenAI-compatible API spec

---

**Last Updated**: 2026-08-26
