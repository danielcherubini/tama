//! Metrics state: counters, system metrics channels, and inference stats.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::proxy::types::{LatestInferenceStats, ProxyMetrics};

/// Aggregated metrics and channel handles for the proxy server.
#[derive(Clone)]
pub(crate) struct MetricsState {
    /// Atomic counters for request/model lifecycle events.
    pub(crate) counters: Arc<ProxyMetrics>,
    /// Latest system-level GPU/CPU/RAM metrics.
    pub(crate) system_metrics: Arc<RwLock<crate::gpu::SystemMetrics>>,
    /// Broadcast channel for per-server metrics snapshots.
    pub(crate) metrics_tx: tokio::sync::broadcast::Sender<crate::gpu::MetricsSnapshot>,
    /// Watch channel for per-backend inference stats (tok/s, cache hit rate, etc.).
    pub(crate) inference_stats: tokio::sync::watch::Sender<HashMap<String, LatestInferenceStats>>,
}

impl MetricsState {
    pub(crate) fn new() -> Self {
        let (metrics_tx, _) = tokio::sync::broadcast::channel(3);
        Self {
            counters: Arc::new(ProxyMetrics::default()),
            system_metrics: Arc::new(RwLock::new(crate::gpu::SystemMetrics::default())),
            metrics_tx,
            inference_stats: tokio::sync::watch::channel(HashMap::new()).0,
        }
    }

    /// Publish a metrics snapshot to all broadcast subscribers.
    pub(crate) fn publish_metrics(&self, snapshot: crate::gpu::MetricsSnapshot) {
        let _ = self.metrics_tx.send(snapshot);
    }

    /// Subscribe to the metrics broadcast channel.
    pub(crate) fn subscribe_metrics(
        &self,
    ) -> tokio::sync::broadcast::Receiver<crate::gpu::MetricsSnapshot> {
        self.metrics_tx.subscribe()
    }

    /// Write a system metrics snapshot into the RwLock.
    pub(crate) async fn set_system_metrics(&self, snapshot: crate::gpu::SystemMetrics) {
        *self.system_metrics.write().await = snapshot;
    }

    /// Read and clone the current system metrics snapshot.
    pub(crate) async fn system_metrics_snapshot(&self) -> crate::gpu::SystemMetrics {
        self.system_metrics.read().await.clone()
    }

    /// Clone the current inference stats HashMap from the watch channel.
    pub(crate) fn inference_stats_snapshot(&self) -> HashMap<String, LatestInferenceStats> {
        self.inference_stats.borrow().clone()
    }

    /// Modify the inference stats map via `send_modify` on the watch channel.
    pub(crate) fn modify_inference_stats(
        &self,
        f: impl FnOnce(&mut HashMap<String, LatestInferenceStats>),
    ) {
        self.inference_stats.send_modify(f);
    }

    /// Clear all inference stats by replacing the watch channel value with an empty HashMap.
    pub(crate) fn clear_inference_stats(&self) {
        let _ = self.inference_stats.send_replace(HashMap::new());
    }

    /// Return a reference to the atomic counters for handler-style bumps.
    #[allow(dead_code)]
    pub(crate) fn counters(&self) -> &Arc<ProxyMetrics> {
        &self.counters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::types::LatestInferenceStats;

    /// Verify that publish_metrics sends a snapshot and subscribe_metrics
    /// receives it on the other end of the broadcast channel.
    #[tokio::test]
    async fn test_publish_subscribe_and_snapshot() {
        let state = MetricsState::new();
        let mut rx = state.subscribe_metrics();

        let snapshot = crate::gpu::MetricsSnapshot {
            buckets: vec![],
            current: crate::gpu::MetricCurrent::default(),
        };
        state.publish_metrics(snapshot);

        let received = rx.recv().await.expect("should receive published snapshot");
        assert!(received.buckets.is_empty());
    }

    /// Verify that set_system_metrics and system_metrics_snapshot round-trip
    /// correctly (write then read back). Also verifies the default is empty.
    #[tokio::test]
    async fn test_set_and_snapshot_system_metrics() {
        let state = MetricsState::new();

        // Default should be all zeros
        let default_snap = state.system_metrics_snapshot().await;
        assert_eq!(default_snap.cpu_usage_pct, 0.0);

        // Write a custom snapshot
        let custom = crate::gpu::SystemMetrics {
            cpu_usage_pct: 42.5,
            ram_used_mib: 8192,
            ..Default::default()
        };
        state.set_system_metrics(custom.clone()).await;

        // Read it back
        let snap = state.system_metrics_snapshot().await;
        assert_eq!(snap.cpu_usage_pct, 42.5);
        assert_eq!(snap.ram_used_mib, 8192);
    }

    /// Verify that clear_inference_stats empties the map.
    #[tokio::test]
    async fn test_clear_inference_stats() {
        let state = MetricsState::new();

        // Seed some stats
        state.modify_inference_stats(|m| {
            m.insert(
                "backend-a".to_string(),
                LatestInferenceStats {
                    tps: Some(10.0),
                    ..Default::default()
                },
            );
            m.insert(
                "backend-b".to_string(),
                LatestInferenceStats {
                    tps: Some(20.0),
                    ..Default::default()
                },
            );
        });
        assert_eq!(state.inference_stats_snapshot().len(), 2);

        // Clear should empty the map
        state.clear_inference_stats();
        let snap = state.inference_stats_snapshot();
        assert!(snap.is_empty());
    }

    /// Verify that counters() returns a reference to the internal counters.
    #[test]
    fn test_counters_returns_arc() {
        let state = MetricsState::new();
        let counters = state.counters();
        // Should be able to read atomic values
        assert_eq!(
            counters
                .total_requests
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    /// Verify that modify_inference_stats applies a closure to the map.
    #[tokio::test]
    async fn test_modify_inference_stats() {
        let state = MetricsState::new();

        // Seed initial stats
        state.modify_inference_stats(|m| {
            m.insert(
                "backend-a".to_string(),
                LatestInferenceStats {
                    tps: Some(10.0),
                    ..Default::default()
                },
            );
        });

        // Modify via closure — change tps for backend-a
        state.modify_inference_stats(|map| {
            if let Some(stats) = map.get_mut("backend-a") {
                stats.tps = Some(99.0);
            }
        });

        // Verify the modification took effect
        let snap = state.inference_stats_snapshot();
        assert_eq!(snap["backend-a"].tps, Some(99.0));
    }
}
