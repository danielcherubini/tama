//! Engine-metrics scraping: Prometheus text parsing, windowed diffing of
//! cumulative counters, and endpoint→/metrics URL rewriting.
//!
//! Pure and dependency-free (`std` + `url`). The HTTP fetching and
//! per-tick budgeting live in [`crate::stats::StatsCollector`]; this
//! module owns the logic that decides what an observation means.
//!
//! The engine is detected from the BODY (one process is one engine
//! binary, so the body's counter prefix identifies the engine):
//!
//! vLLM (served by default at `/metrics`):
//!   `vllm:generation_tokens_total`
//!   `vllm:prompt_tokens_by_source_total{source="local_compute"}`
//!   `vllm:prompt_tokens_by_source_total{source="local_cache_hit"}`
//!   `vllm:spec_decode_num_drafts_total`
//!   `vllm:spec_decode_num_draft_tokens_total`
//!   `vllm:spec_decode_num_accepted_tokens_total`
//!
//! llama.cpp (served at `/metrics` only with `--metrics`):
//!   `llamacpp:tokens_predicted_total`
//!   `llamacpp:prompt_tokens_total` (already excludes cache hits upstream)
//!   `llamacpp:prompt_tokens_cached_total`
//!   `llamacpp:spec_decode_num_drafts_total`
//!   `llamacpp:spec_decode_num_draft_tokens_total`
//!   `llamacpp:spec_decode_num_accepted_tokens_total`

use url::Url;

/// How often each endpoint is re-scraped (vLLM logs its spec summary every
/// 10s, so 10s gives one observation per backend log line).
pub const SCRAPE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);
/// The single staleness clock (ADR-0014): an observation older than 30s
/// reads as inactive — renames, restarts, or a wedged scrape silence the
/// values instead of showing a stale rate forever.
pub const STALE_MS: i64 = 30_000;
/// Per-endpoint scrape timeout; a wedged /metrics must not stall the tick.
pub const PER_SCRAPE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
/// Cumulative scrape budget per tick: the tick must never linger, or the
/// proxy's 5s `LIVE_FRAME_MAX_AGE` freshness gate blanks every model on
/// the host. Skipped models simply retry next tick.
pub const TICK_SCRAPE_BUDGET: std::time::Duration = std::time::Duration::from_secs(3);

/// Cumulative counters summed across all label sets, per engine prefix.
/// A slot is `None` when the engine's body doesn't expose that counter
/// (e.g. pre-2026 vLLM has no `prompt_tokens_by_source`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineCounters {
    /// vllm:generation_tokens_total / llamacpp:tokens_predicted_total
    pub decode_tokens: Option<f64>,
    /// vllm:prompt_tokens_by_source_total{source="local_compute"} (compute-only;
    /// None on pre-2026 vLLM) / llamacpp:prompt_tokens_total (already excludes
    /// cache hits upstream)
    pub prompt_computed: Option<f64>,
    /// vllm:prompt_tokens_by_source_total{source="local_cache_hit"}
    /// / llamacpp:prompt_tokens_cached_total
    pub prompt_cached: Option<f64>,
    /// vllm:spec_decode_num_drafts_total / llamacpp:spec_decode_num_drafts_total
    pub spec_drafts: Option<f64>,
    /// vllm:spec_decode_num_draft_tokens_total / llamacpp:spec_decode_num_draft_tokens_total
    pub spec_draft_tokens: Option<f64>,
    /// vllm:spec_decode_num_accepted_tokens_total / llamacpp:spec_decode_num_accepted_tokens_total
    pub spec_accepted_tokens: Option<f64>,
}

/// Which engine family the scraped body belongs to (body-driven detection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Vllm,
    Llamacpp,
}

/// Parse the engine counters out of a Prometheus text exposition.
///
/// Skips blank lines, `#` comment lines (HELP/TYPE/EOF), and any line
/// whose value token is not a finite `f64` — while keeping the rest of
/// the body. Label values containing spaces are parsed (the name is
/// recovered as the prefix before the first `{`), and a trailing
/// Prometheus timestamp is ignored whenever a finite value precedes
/// it. Sums the value across ALL label sets for each exact counter name
/// — names are matched exactly, so a metric with a longer suffix
/// (e.g. `..._drafts_total_extra`) is never counted.
///
/// Kind detection is body-driven: any of the vLLM table names matched →
/// `Vllm`; else any of the llama.cpp table names matched → `Llamacpp`;
/// else `None` (unknown engine — the no-op path). Both prefixes present:
/// `Vllm` wins (documented non-case: one process is one engine binary).
pub fn parse_engine_metrics(body: &str) -> Option<(EngineKind, EngineCounters)> {
    let mut c = EngineCounters::default();
    let mut vllm_seen = [false; 5];
    let mut lmcpp_seen = [false; 6];
    for line in body.lines() {
        let line = line.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value, labels)) = split_metric_line(line) else {
            continue;
        };
        let source = labels.as_deref().and_then(source_label_value);
        match name.as_str() {
            "vllm:generation_tokens_total" => {
                vllm_seen[0] = true;
                c.decode_tokens = Some(c.decode_tokens.unwrap_or(0.0) + value);
            }
            "vllm:prompt_tokens_by_source_total" => {
                vllm_seen[1] = true;
                match source {
                    Some("local_compute") => {
                        c.prompt_computed = Some(c.prompt_computed.unwrap_or(0.0) + value);
                    }
                    Some("local_cache_hit") => {
                        c.prompt_cached = Some(c.prompt_cached.unwrap_or(0.0) + value);
                    }
                    _ => {}
                }
            }
            "vllm:spec_decode_num_drafts_total" => {
                vllm_seen[2] = true;
                c.spec_drafts = Some(c.spec_drafts.unwrap_or(0.0) + value);
            }
            "vllm:spec_decode_num_draft_tokens_total" => {
                vllm_seen[3] = true;
                c.spec_draft_tokens = Some(c.spec_draft_tokens.unwrap_or(0.0) + value);
            }
            "vllm:spec_decode_num_accepted_tokens_total" => {
                vllm_seen[4] = true;
                c.spec_accepted_tokens = Some(c.spec_accepted_tokens.unwrap_or(0.0) + value);
            }
            "llamacpp:tokens_predicted_total" => {
                lmcpp_seen[0] = true;
                c.decode_tokens = Some(c.decode_tokens.unwrap_or(0.0) + value);
            }
            "llamacpp:prompt_tokens_total" => {
                lmcpp_seen[1] = true;
                c.prompt_computed = Some(c.prompt_computed.unwrap_or(0.0) + value);
            }
            "llamacpp:prompt_tokens_cached_total" => {
                lmcpp_seen[2] = true;
                c.prompt_cached = Some(c.prompt_cached.unwrap_or(0.0) + value);
            }
            "llamacpp:spec_decode_num_drafts_total" => {
                lmcpp_seen[3] = true;
                c.spec_drafts = Some(c.spec_drafts.unwrap_or(0.0) + value);
            }
            "llamacpp:spec_decode_num_draft_tokens_total" => {
                lmcpp_seen[4] = true;
                c.spec_draft_tokens = Some(c.spec_draft_tokens.unwrap_or(0.0) + value);
            }
            "llamacpp:spec_decode_num_accepted_tokens_total" => {
                lmcpp_seen[5] = true;
                c.spec_accepted_tokens = Some(c.spec_accepted_tokens.unwrap_or(0.0) + value);
            }
            _ => {}
        }
    }
    if vllm_seen.iter().any(|s| *s) {
        Some((EngineKind::Vllm, c))
    } else if lmcpp_seen.iter().any(|s| *s) {
        Some((EngineKind::Llamacpp, c))
    } else {
        None
    }
}

/// Split a `name{...maybe labels...} <value>` line into its exact metric
/// name, value, and the raw label-set text (the content between the first
/// `{` and the first `}` that is NOT inside a quoted value — a legal
/// Prometheus label value may contain `}` inside quotes; if no closing
/// `}` exists, the label set is unparseable and the line is treated as
/// having no labels).
///
/// Guarantees, over the whitespace-token sequence `t[0..n]` (`n >= 2`,
/// else `None`):
///
/// - **Spaced label values are accepted.** A quoted label value containing
///   a space (`name{model_name="foo bar"} 5`) splits the line into extra
///   tokens, but the name is the single-space rejoin of the tokens
///   before the value token, truncated at the FIRST `{` — so the exact,
///   space-free counter name always lives entirely in `t[0]` and
///   rejoining can never mix label text into it.
/// - **Trailing timestamps are ignored, never read as the value.** When
///   the last token is a plain integer and the second-to-last parses to
///   a finite `f64`, the second-to-last is the value (the last token is
///   treated as an ignorable trailing Prometheus timestamp).
/// - **Non-finite values are rejected.** Any value token that parses to
///   NaN / ±Inf disqualifies the line, since it would otherwise flow
///   into an observation and surface as a "NaN%" card.
///
/// Degenerate case (legal Prometheus has at most one token after the
/// value): on a 4-token line whose last token is a plain integer and
/// second-to-last a finite value, the second-to-last token is accepted
/// as the value — `name 5.0 1700000000000 22` yields `1700000000000`.
/// Real expositions never produce this; the exact-name match in the
/// caller bounds the blast radius of such malformed input.
///
/// Returns `None` for any line that does not yield a finite value.
fn split_metric_line(line: &str) -> Option<(String, f64, Option<String>)> {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    let n = tokens.len();
    if n < 2 {
        return None;
    }
    // Pick the value token: a plain-integer last token after a finite
    // value is a trailing timestamp — use the token before it instead.
    // Otherwise the value is the last token.
    let value_idx = if n >= 3
        && is_plain_int(tokens[n - 1])
        && tokens[n - 2]
            .parse::<f64>()
            .ok()
            .is_some_and(|v: f64| v.is_finite())
    {
        n - 2
    } else {
        n - 1
    };
    let value: f64 = tokens[value_idx].parse().ok()?;
    // NaN / ±Inf must not reach an observation — reject the line.
    if !value.is_finite() {
        return None;
    }
    // The tokens before the value form `metric{...labels...}` (a label
    // value may contain spaces, so rejoin with single spaces), then
    // truncate at the first `{` where the label set begins.
    let rejoined = tokens[..value_idx].join(" ");
    let (name, labels) = match rejoined.find('{') {
        Some(i) => {
            let after = &rejoined[i + 1..];
            // The label set ends at the first `}` that is NOT inside a
            // quoted value (a legal label value may contain `}`).
            let mut in_quotes = false;
            let mut end = None;
            for (j, b) in after.bytes().enumerate() {
                if b == b'"' {
                    in_quotes = !in_quotes;
                } else if b == b'}' && !in_quotes {
                    end = Some(j);
                    break;
                }
            }
            match end {
                Some(j) => (rejoined[..i].to_string(), Some(after[..j].to_string())),
                None => (rejoined[..i].to_string(), None),
            }
        }
        None => (rejoined, None),
    };
    Some((name, value, labels))
}

/// Read the `source` label's value out of a raw label set
/// (`k1="v1",k2="v2"`). The key is anchored — it must be EXACTLY
/// `source` at the start of the set or immediately after a `,` outside
/// a quoted value (so e.g. `other_source="local_compute"` never matches).
/// The value is the quoted string following `=`, or the unquoted token
/// (legal for numbers / NaN / Inf).
fn source_label_value(labels: &str) -> Option<&str> {
    let mut pos = 0;
    loop {
        let rest = &labels[pos..];
        if let Some(after) = rest.strip_prefix("source=") {
            return read_label_value(after);
        }
        // Advance to the next `,` outside a quoted value; the key after
        // it is the next candidate.
        let mut i = 0;
        let mut in_quotes = false;
        while i < rest.len() {
            let b = rest.as_bytes()[i];
            if b == b'"' {
                in_quotes = !in_quotes;
            } else if b == b',' && !in_quotes {
                break;
            }
            i += 1;
        }
        if i == rest.len() {
            return None;
        }
        pos += i + 1;
    }
}

/// Read a label value following `=`: the quoted string, or (legal for
/// numbers / NaN / Inf) the unquoted token up to the next comma or
/// whitespace. Asymmetry with the quoted form: the unquoted value
/// requires a terminating `,` or whitespace, so a final unquoted label
/// (e.g. `source=5` at end-of-string) yields `None` rather than `"5"`.
/// Harmless here — only the quoted `source` values are matched.
fn read_label_value(s: &str) -> Option<&str> {
    if let Some(v) = s.strip_prefix('"') {
        let end = v.find('"')?;
        Some(&v[..end])
    } else {
        let end = s.find(|c: char| c == ',' || c.is_ascii_whitespace())?;
        Some(&s[..end])
    }
}

/// Whether `tok` is a plain integer (`-?[0-9]+`) — the shape of a
/// trailing Prometheus timestamp.
fn is_plain_int(tok: &str) -> bool {
    let digits = tok.strip_prefix('-').unwrap_or(tok);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// One scrape window's observation. All `None` except `had_traffic` means
/// "no traffic in this window" (the caller keeps the last observation).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindowObs {
    pub tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub cache_hit_pct: Option<f64>,
    pub spec_accept_pct: Option<f64>,
    /// Window had spec-decoding traffic (Δdraft_tokens > 0).
    pub spec_active: bool,
    pub had_traffic: bool,
}

/// Windowed rates from two successive cumulative counter sets.
///
/// - `prev` is `None` → `None` (first scrape — no delta to compute)
/// - any counter present in both `prev` and `cur` that DECREASED → `None`
///   (engine restart / counter reset — never emit a bogus rate)
/// - no counter advanced (all Δ ≤ 0) → `None` (idle window — the caller
///   keeps the last observation until it goes stale)
/// - otherwise `Some(WindowObs)`:
///   - `tps` = Δdecode_tokens / dt_secs when the counter is present in both
///     (Δ may be 0.0 — a prefill-only window), else `None`
///   - `prompt_tps` = Δprompt_computed / dt_secs (same availability rule)
///   - `cache_hit_pct` = 100 × Δprompt_cached / (Δprompt_cached + Δprompt_computed)
///     when the denominator > 0, else `None` (disjoint-set semantics, matching
///     the retired per-response cache_n/prompt_n)
///   - `spec_accept_pct` = 100 × Δaccepted / Δdraft_tokens when Δdraft_tokens > 0
///   - `spec_active` = Δdraft_tokens > 0
///   - `had_traffic` = true
/// - `dt_secs <= 0.0` → `None` (defensive; the 10s throttle makes this unreachable
///   in production)
pub fn observe(
    prev: Option<EngineCounters>,
    cur: &EngineCounters,
    dt_secs: f64,
) -> Option<WindowObs> {
    let prev = prev?;
    if dt_secs <= 0.0 {
        return None;
    }
    // Any counter present in both sets that DECREASED → engine restart /
    // counter reset → discard the whole window.
    for (p, c) in [
        (prev.decode_tokens, cur.decode_tokens),
        (prev.prompt_computed, cur.prompt_computed),
        (prev.prompt_cached, cur.prompt_cached),
        (prev.spec_drafts, cur.spec_drafts),
        (prev.spec_draft_tokens, cur.spec_draft_tokens),
        (prev.spec_accepted_tokens, cur.spec_accepted_tokens),
    ] {
        if let (Some(p), Some(c)) = (p, c) {
            if c < p {
                return None;
            }
        }
    }
    let d_decode = match (prev.decode_tokens, cur.decode_tokens) {
        (Some(p), Some(c)) => Some(c - p),
        _ => None,
    };
    let d_computed = match (prev.prompt_computed, cur.prompt_computed) {
        (Some(p), Some(c)) => Some(c - p),
        _ => None,
    };
    let d_cached = match (prev.prompt_cached, cur.prompt_cached) {
        (Some(p), Some(c)) => Some(c - p),
        _ => None,
    };
    let d_draft_tokens = match (prev.spec_draft_tokens, cur.spec_draft_tokens) {
        (Some(p), Some(c)) => Some(c - p),
        _ => None,
    };
    let d_accepted = match (prev.spec_accepted_tokens, cur.spec_accepted_tokens) {
        (Some(p), Some(c)) => Some(c - p),
        _ => None,
    };
    // Idle window: no counter advanced → no observation (the caller keeps
    // the last one until it goes stale).
    let advanced = [d_decode, d_computed, d_cached, d_draft_tokens, d_accepted]
        .iter()
        .any(|d| d.is_some_and(|v| v > 0.0));
    if !advanced {
        return None;
    }
    let spec_active = d_draft_tokens.is_some_and(|v| v > 0.0);
    Some(WindowObs {
        tps: d_decode.map(|d| d / dt_secs),
        prompt_tps: d_computed.map(|d| d / dt_secs),
        cache_hit_pct: match (d_cached, d_computed) {
            (Some(c), Some(p)) if c + p > 0.0 => Some(100.0 * c / (c + p)),
            _ => None,
        },
        spec_accept_pct: match (d_accepted, d_draft_tokens) {
            (Some(a), Some(d)) if d > 0.0 => Some(100.0 * a / d),
            _ => None,
        },
        spec_active,
        had_traffic: true,
    })
}

/// Rewrite an engine endpoint URL to its `/metrics` path:
/// `"http://127.0.0.1:8000/v1"` → `Some("http://127.0.0.1:8000/metrics")`.
/// A bare `http://host:9000` (no path) works as-is; `https` is preserved;
/// any non-http(s) scheme (e.g. `grpc://`) → `None`.
pub fn metrics_url_for(endpoint_url: &str) -> Option<String> {
    let parsed = Url::parse(endpoint_url).ok()?;
    match parsed.scheme() {
        "http" | "https" => {}
        _ => return None,
    }
    let mut u = parsed;
    u.set_path("/metrics");
    u.set_query(None);
    u.set_fragment(None);
    Some(u.to_string())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic vLLM /metrics body: comments, unrelated metrics, labels
    /// with `=` and spaces in values, and the three spec counters each in
    /// two label sets. Values must sum across label sets. The raw
    /// `vllm:prompt_tokens_total` is NOT a table name — it must not leak
    /// into any slot.
    const VLLM_BODY: &str = r#"# HELP vllm:prompt_tokens_total Total number of prompt tokens processed
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="Qwen3-30B-A3B",engine="0"} 12345.0
vllm:num_requests_running{model_name="Qwen3-30B-A3B",engine="0"} 1.0
# HELP vllm:spec_decode_num_drafts_total Total number of spec decoding iterations
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="Qwen3-30B-A3B",engine="0"} 57.0
vllm:spec_decode_num_drafts_total{model_name="Qwen3-30B-A3B",engine="1"} 58.0
vllm:spec_decode_num_draft_tokens_total{model_name="Qwen3-30B-A3B",engine="0"} 185.5
vllm:spec_decode_num_draft_tokens_total{model_name="Qwen3-30B-A3B",engine="1"} 186.0
vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen3-30B-A3B",engine="0"} 82.0
vllm:spec_decode_num_accepted_tokens_total{model_name="Qwen3-30B-A3B",engine="1"} 84.0
vllm:kv_cache_usage_perc{model_name="Qwen3-30B-A3B",engine="0"} 0.42
http_requests_total{method="GET",uri="/v1/chat/completions",ok="true"} 99.0

# EOF
"#;

    /// All slots present at zero — the "first scrape saw the counters"
    /// baseline.
    fn zero_counters() -> EngineCounters {
        EngineCounters {
            decode_tokens: Some(0.0),
            prompt_computed: Some(0.0),
            prompt_cached: Some(0.0),
            spec_drafts: Some(0.0),
            spec_draft_tokens: Some(0.0),
            spec_accepted_tokens: Some(0.0),
        }
    }

    #[test]
    fn test_parse_vllm_two_label_sets_sum() {
        let (kind, c) = parse_engine_metrics(VLLM_BODY).expect("three counters present");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(
            c,
            EngineCounters {
                decode_tokens: None,
                prompt_computed: None,
                prompt_cached: None,
                spec_drafts: Some(115.0),
                spec_draft_tokens: Some(371.5),
                spec_accepted_tokens: Some(166.0),
            }
        );
    }

    /// A body without any of the table names → `None`, even with a
    /// tokenizing label and even with the `vllm:` prefix — the prefix
    /// alone is not a table name.
    #[test]
    fn test_parse_missing_counters_none() {
        let body = "\
vllm:prompt_tokens_total{model_name=\"x\",engine=\"0\"} 1.0\n\
llamacpp_duration_s{status_stage=\"0\",lifespan_stage=\"0\",vram_stage=\"3\"} 2.5\n\
process_cpu_seconds_total 42.0\n";
        assert_eq!(parse_engine_metrics(body), None);
    }

    /// The rest of the body still parses when a line's trailing value token
    /// is unparseable — that one line is skipped, the others are kept.
    #[test]
    fn test_parse_unparseable_value_line_skipped() {
        let body = "\
vllm:spec_decode_num_drafts_total{model_name=\"x\",engine=\"0\"} 10.0\nbroken_metric{a=\"b\"} not_a_number\nvllm:spec_decode_num_draft_tokens_total{model_name=\"x\",engine=\"0\"} 30.0\nvllm:spec_decode_num_accepted_tokens_total{model_name=\"x\",engine=\"0\"} 15.0\n";
        let (kind, c) = parse_engine_metrics(body).expect("two valid lines remain");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(
            c,
            EngineCounters {
                decode_tokens: None,
                prompt_computed: None,
                prompt_cached: None,
                spec_drafts: Some(10.0),
                spec_draft_tokens: Some(30.0),
                spec_accepted_tokens: Some(15.0),
            }
        );
    }

    /// Exact-name guard: a longer-suffixed counter must NOT match.
    #[test]
    fn test_parse_exact_name_guard() {
        let body = "vllm:spec_decode_num_drafts_total_extra{model_name=\"x\",engine=\"0\"} 7.0\n";
        assert_eq!(parse_engine_metrics(body), None);
    }

    /// A label set whose quoted value contains a space is a legal
    /// Prometheus form (plan Task 2) and MUST be parsed: the name is the
    /// prefix before the first `{`, the value is the last finite token.
    /// Values are summed across all label sets for the counter.
    #[test]
    fn test_parse_labels_with_spaces() {
        let body = "\
            vllm:spec_decode_num_drafts_total{model_name=\"Instruct Model\",engine=\"0\"} 3.0\n\
            vllm:spec_decode_num_drafts_total{model_name=\"Instruct Model\",engine=\"1\"} 4.0\n\
            vllm:spec_decode_num_draft_tokens_total{model_name=\"a b\",engine=\"0\"} 100\n\
            vllm:spec_decode_num_accepted_tokens_total{model_name=\"a b\",engine=\"0\"} 60\n";
        let (kind, c) = parse_engine_metrics(body).expect("spaced label lines parse");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(
            c,
            EngineCounters {
                decode_tokens: None,
                prompt_computed: None,
                prompt_cached: None,
                spec_drafts: Some(7.0),
                spec_draft_tokens: Some(100.0),
                spec_accepted_tokens: Some(60.0),
            }
        );
    }

    /// A trailing Prometheus timestamp must never be read as the counter
    /// value. When a finite value precedes a trailing plain-integer token,
    /// the value is used and the timestamp ignored — including when the
    /// label set itself carries a spaced quoted value.
    #[test]
    fn test_parse_trailing_timestamp_ignored() {
        // No label set: timestamp ignored, real value used.
        let (kind, c) =
            parse_engine_metrics("vllm:spec_decode_num_drafts_total 57.0 1700000000000")
                .expect("value precedes the timestamp");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(c.spec_drafts, Some(57.0), "timestamp must not be the value");

        // Labeled form (realistic vLLM output): same rule.
        let (_, c) = parse_engine_metrics(
            r#"vllm:spec_decode_num_drafts_total{model_name="x",engine="0"} 5.0 1700000000000"#,
        )
        .expect("labeled line parses");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(c.spec_drafts, Some(5.0), "timestamp must not be the value");

        // Spaced label value plus timestamp: value used, timestamp ignored.
        let (_, c) = parse_engine_metrics(
            r#"vllm:spec_decode_num_drafts_total{model_name="a b",engine="0"} 5.0 1700000000000"#,
        )
        .expect("spaced label + timestamp parses");
        assert_eq!(
            c.spec_drafts,
            Some(5.0),
            "timestamp must not be summed into the value"
        );

        // Mixed: each line counts at its genuine value.
        let mixed = r#"vllm:spec_decode_num_drafts_total{model_name="x",engine="0"} 4.0
vllm:spec_decode_num_drafts_total{model_name="x",engine="1"} 100.0 1700000000000"#;
        let (_, c) = parse_engine_metrics(mixed).expect("both lines count at their values");
        assert_eq!(
            c.spec_drafts,
            Some(104.0),
            "timestamp must not be summed in"
        );
    }

    /// A non-finite value token (NaN / ±Inf) is rejected: a `NaN`
    /// counter must not flow into an observation (it would surface as a
    /// "NaN%" card downstream).
    #[test]
    fn test_parse_non_finite_value_skipped() {
        let nan = r#"vllm:spec_decode_num_drafts_total{model_name="x",engine="0"} NaN"#;
        assert_eq!(parse_engine_metrics(nan), None);
        let inf = r#"vllm:spec_decode_num_drafts_total{model_name="x",engine="0"} inf"#;
        assert_eq!(parse_engine_metrics(inf), None);

        // Mixed: the NaN line is dropped, the other counters survive.
        let mixed = r#"vllm:spec_decode_num_drafts_total{model_name="x",engine="0"} 4.0
vllm:spec_decode_num_drafts_total{model_name="x",engine="1"} NaN
vllm:spec_decode_num_draft_tokens_total{model_name="x",engine="0"} 8.0
vllm:spec_decode_num_accepted_tokens_total{model_name="x",engine="0"} 2.0"#;
        let (_, c) = parse_engine_metrics(mixed).expect("remaining counters survive");
        assert_eq!(c.spec_drafts, Some(4.0));
        assert!(c.spec_drafts.unwrap().is_finite());
    }

    /// An unlabelled llama.cpp /metrics body: all six table counters
    /// populate their slots (kind `Llamacpp`), and the longer-suffixed
    /// seconds counter must NOT count.
    #[test]
    fn test_parse_llamacpp_body() {
        let body = "\
# HELP llamacpp:tokens_predicted_total Total number of tokens predicted
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 12345.0
llamacpp:tokens_predicted_seconds_total 12.5
llamacpp:prompt_tokens_total 2000.0
llamacpp:prompt_tokens_cached_total 500.0
llamacpp:spec_decode_num_drafts_total 57.0
llamacpp:spec_decode_num_draft_tokens_total 185.0
llamacpp:spec_decode_num_accepted_tokens_total 82.0
";
        let (kind, c) = parse_engine_metrics(body).expect("llama.cpp counters present");
        assert_eq!(kind, EngineKind::Llamacpp);
        assert_eq!(
            c,
            EngineCounters {
                decode_tokens: Some(12345.0),
                prompt_computed: Some(2000.0),
                prompt_cached: Some(500.0),
                spec_drafts: Some(57.0),
                spec_draft_tokens: Some(185.0),
                spec_accepted_tokens: Some(82.0),
            }
        );
    }

    /// A body carrying BOTH prefixes: one process is one engine binary, so
    /// `Vllm` wins (documented non-case).
    #[test]
    fn test_parse_mixed_prefixes_prefers_vllm() {
        let body = "\
vllm:generation_tokens_total{model_name=\"m\",engine=\"0\"} 100.0\n\
llamacpp:tokens_predicted_total 50.0\n\
llamacpp:prompt_tokens_total 10.0\n";
        let (kind, c) = parse_engine_metrics(body).expect("both prefixes present");
        assert_eq!(
            kind,
            EngineKind::Vllm,
            "vLLM wins the mixed-prefix non-case"
        );
        assert_eq!(c.decode_tokens, Some(150.0));
        assert_eq!(c.prompt_computed, Some(10.0));
    }

    /// The `by_source` slots filter on the `source` label: only
    /// `local_compute` and `local_cache_hit` count (into their slots,
    /// summed across label sets); a sibling value like
    /// `external_kv_transfer` and a non-`source` key never count.
    #[test]
    fn test_parse_by_source_labels() {
        let body = "\
vllm:prompt_tokens_by_source_total{source=\"local_compute\",model_name=\"m\",engine=\"0\"} 100.0\n\
vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\",model_name=\"m\",engine=\"0\"} 50.0\n\
vllm:prompt_tokens_by_source_total{source=\"external_kv_transfer\",model_name=\"m\",engine=\"0\"} 7.0\n\
vllm:prompt_tokens_by_source_total{source=\"local_compute\",model_name=\"n\",engine=\"0\"} 25.0\n\
vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\",model_name=\"n\",engine=\"0\"} 5.0\n\
vllm:prompt_tokens_by_source_total{other_source=\"local_compute\",model_name=\"m\",engine=\"0\"} 9.0\n";
        let (kind, c) = parse_engine_metrics(body).expect("by_source lines present");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(
            c.prompt_computed,
            Some(125.0),
            "local_compute label sets only"
        );
        assert_eq!(
            c.prompt_cached,
            Some(55.0),
            "local_cache_hit label sets only"
        );
        assert_eq!(c.decode_tokens, None);
        assert_eq!(c.spec_drafts, None);
    }

    /// Pre-2026 vLLM: `prompt_tokens_by_source` doesn't exist yet —
    /// `prompt_computed`/`prompt_cached` are `None`, the other four slots
    /// populate, kind `Vllm` (the raw `prompt_tokens_total` is NOT a
    /// table name and must not leak into a slot).
    #[test]
    fn test_parse_pre_2026_vllm_no_by_source() {
        let body = "\
vllm:prompt_tokens_total{model_name=\"m\",engine=\"0\"} 12345.0\n\
vllm:generation_tokens_total{model_name=\"m\",engine=\"0\"} 999.0\n\
vllm:spec_decode_num_drafts_total{model_name=\"m\",engine=\"0\"} 57.0\n\
vllm:spec_decode_num_draft_tokens_total{model_name=\"m\",engine=\"0\"} 185.0\n\
vllm:spec_decode_num_accepted_tokens_total{model_name=\"m\",engine=\"0\"} 82.0\n";
        let (kind, c) = parse_engine_metrics(body).expect("vLLM counters present");
        assert_eq!(kind, EngineKind::Vllm);
        assert_eq!(
            c,
            EngineCounters {
                decode_tokens: Some(999.0),
                prompt_computed: None,
                prompt_cached: None,
                spec_drafts: Some(57.0),
                spec_draft_tokens: Some(185.0),
                spec_accepted_tokens: Some(82.0),
            }
        );
    }

    #[test]
    fn test_observe_first_scrape_none() {
        let cur = EngineCounters {
            decode_tokens: Some(500.0),
            prompt_computed: Some(200.0),
            prompt_cached: Some(50.0),
            spec_drafts: Some(115.0),
            spec_draft_tokens: Some(371.0),
            spec_accepted_tokens: Some(165.0),
        };
        assert_eq!(observe(None, &cur, 10.0), None);
    }

    /// Real-log vector: window counters "Accepted: 165, Drafted: 371" ⇒
    /// "Avg Draft acceptance rate: 44.5%".
    #[test]
    fn test_observe_real_log_window() {
        let prev = zero_counters();
        let cur = EngineCounters {
            decode_tokens: None,
            prompt_computed: None,
            prompt_cached: None,
            spec_drafts: Some(115.0),
            spec_draft_tokens: Some(371.0),
            spec_accepted_tokens: Some(165.0),
        };
        let obs = observe(Some(prev), &cur, 10.0).expect("expected a rate");
        assert!(obs.spec_active);
        assert!(
            (obs.spec_accept_pct.unwrap() - 44.47).abs() < 0.01,
            "got {obs:?}"
        );
        assert_eq!(obs.tps, None, "decode counter unavailable");
    }

    /// Intermediate window: (200-133)/(450-300) = 67/150 ≈ 44.667.
    #[test]
    fn test_observe_intermediate_window() {
        let prev = EngineCounters {
            decode_tokens: None,
            prompt_computed: None,
            prompt_cached: None,
            spec_drafts: Some(100.0),
            spec_draft_tokens: Some(300.0),
            spec_accepted_tokens: Some(133.0),
        };
        let cur = EngineCounters {
            decode_tokens: None,
            prompt_computed: None,
            prompt_cached: None,
            spec_drafts: Some(140.0),
            spec_draft_tokens: Some(450.0),
            spec_accepted_tokens: Some(200.0),
        };
        let obs = observe(Some(prev), &cur, 10.0).expect("expected a rate");
        assert!(
            (obs.spec_accept_pct.unwrap() - 44.6666667).abs() < 0.001,
            "got {obs:?}"
        );
    }

    /// Explicit `dt_secs = 10.0` window over the full counter set:
    /// every rate is the windowed Δ/Δt (the cache ratio is the
    /// disjoint-set share, 50/(50+200) = 20%).
    #[test]
    fn test_observe_window_rates() {
        let prev = zero_counters();
        let cur = EngineCounters {
            decode_tokens: Some(500.0),
            prompt_computed: Some(200.0),
            prompt_cached: Some(50.0),
            spec_drafts: Some(115.0),
            spec_draft_tokens: Some(100.0),
            spec_accepted_tokens: Some(45.0),
        };
        let obs = observe(Some(prev), &cur, 10.0).expect("window had traffic");
        assert_eq!(obs.tps, Some(50.0));
        assert_eq!(obs.prompt_tps, Some(20.0));
        assert_eq!(obs.cache_hit_pct, Some(20.0));
        assert_eq!(obs.spec_accept_pct, Some(45.0));
        assert!(obs.spec_active);
        assert!(obs.had_traffic);
    }

    /// cur == prev → no counter advanced → idle window → `None`.
    #[test]
    fn test_observe_idle_window_none() {
        let c = EngineCounters {
            decode_tokens: Some(500.0),
            prompt_computed: Some(200.0),
            prompt_cached: Some(50.0),
            spec_drafts: Some(115.0),
            spec_draft_tokens: Some(371.0),
            spec_accepted_tokens: Some(165.0),
        };
        assert_eq!(observe(Some(c), &c, 10.0), None);
    }

    /// Any counter present in both sets that DECREASED → engine restart
    /// / counter reset → `None` (never emit a bogus rate).
    #[test]
    fn test_observe_reset_none() {
        let base_prev = EngineCounters {
            decode_tokens: Some(500.0),
            prompt_computed: Some(200.0),
            prompt_cached: Some(50.0),
            spec_drafts: Some(115.0),
            spec_draft_tokens: Some(371.0),
            spec_accepted_tokens: Some(165.0),
        };
        let reset_decode = EngineCounters {
            decode_tokens: Some(400.0),
            prompt_computed: Some(250.0),
            prompt_cached: Some(60.0),
            spec_drafts: Some(120.0),
            spec_draft_tokens: Some(400.0),
            spec_accepted_tokens: Some(180.0),
        };
        let reset_computed = EngineCounters {
            decode_tokens: Some(600.0),
            prompt_computed: Some(100.0),
            prompt_cached: Some(60.0),
            spec_drafts: Some(120.0),
            spec_draft_tokens: Some(400.0),
            spec_accepted_tokens: Some(180.0),
        };
        let reset_cached = EngineCounters {
            decode_tokens: Some(600.0),
            prompt_computed: Some(250.0),
            prompt_cached: Some(40.0),
            spec_drafts: Some(120.0),
            spec_draft_tokens: Some(400.0),
            spec_accepted_tokens: Some(180.0),
        };
        let reset_spec = EngineCounters {
            decode_tokens: Some(600.0),
            prompt_computed: Some(250.0),
            prompt_cached: Some(60.0),
            spec_drafts: Some(120.0),
            spec_draft_tokens: Some(300.0),
            spec_accepted_tokens: Some(180.0),
        };
        assert_eq!(observe(Some(base_prev), &reset_decode, 10.0), None);
        assert_eq!(observe(Some(base_prev), &reset_computed, 10.0), None);
        assert_eq!(observe(Some(base_prev), &reset_cached, 10.0), None);
        assert_eq!(observe(Some(base_prev), &reset_spec, 10.0), None);
    }

    /// A counter `None` in `cur` is simply unavailable — the other slots
    /// still get their windowed rates.
    #[test]
    fn test_observe_partial_availability() {
        let prev = EngineCounters {
            decode_tokens: Some(100.0),
            prompt_computed: Some(50.0),
            prompt_cached: None,
            spec_drafts: None,
            spec_draft_tokens: None,
            spec_accepted_tokens: None,
        };
        let cur = EngineCounters {
            decode_tokens: Some(300.0),
            prompt_computed: None,
            prompt_cached: None,
            spec_drafts: None,
            spec_draft_tokens: None,
            spec_accepted_tokens: None,
        };
        let obs = observe(Some(prev), &cur, 10.0).expect("decode advanced");
        assert_eq!(obs.tps, Some(20.0));
        assert_eq!(obs.prompt_tps, None);
        assert_eq!(obs.cache_hit_pct, None);
        assert_eq!(obs.spec_accept_pct, None);
        assert!(!obs.spec_active);
    }

    /// `dt_secs <= 0.0` → `None` (defensive; the 10s throttle makes this
    /// unreachable in production).
    #[test]
    fn test_observe_zero_dt_none() {
        let prev = zero_counters();
        let cur = EngineCounters {
            decode_tokens: Some(500.0),
            prompt_computed: Some(200.0),
            prompt_cached: Some(50.0),
            spec_drafts: Some(115.0),
            spec_draft_tokens: Some(100.0),
            spec_accepted_tokens: Some(45.0),
        };
        assert_eq!(observe(Some(prev), &cur, 0.0), None);
    }

    #[test]
    fn test_metrics_url_with_path() {
        assert_eq!(
            metrics_url_for("http://127.0.0.1:8000/v1"),
            Some("http://127.0.0.1:8000/metrics".to_string())
        );
    }

    #[test]
    fn test_metrics_url_bare_host() {
        assert_eq!(
            metrics_url_for("http://host:9000"),
            Some("http://host:9000/metrics".to_string())
        );
    }

    #[test]
    fn test_metrics_url_https_preserved() {
        assert_eq!(
            metrics_url_for("https://inhost:8000/v1"),
            Some("https://inhost:8000/metrics".to_string())
        );
    }

    #[test]
    fn test_metrics_url_non_http_scheme() {
        assert_eq!(metrics_url_for("grpc://x"), None);
    }
}
