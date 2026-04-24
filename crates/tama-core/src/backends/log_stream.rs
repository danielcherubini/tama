//! Backend log streaming — broadcasts backend stdout/stderr lines to SSE subscribers.
//!
//! Each backend instance gets its own `BackendLogStream` (identified by the server name).
//! Lines are stored in a ring buffer and broadcast via a tokio broadcast channel.

use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

/// Maximum number of log lines to retain for replay on reconnect.
const LOG_HEAD_CAP: usize = 200;

/// Broadcast channel capacity — prevents backpressure from many SSE subscribers.
const LOG_BROADCAST_CAP: usize = 1024;

/// A single backend's log stream.
pub struct BackendLogStream {
    /// Oldest lines (always replayed on reconnect).
    head: RwLock<VecDeque<String>>,
    /// Recent lines (after head is full).
    tail: RwLock<VecDeque<String>>,
    /// Broadcast channel for live delivery.
    tx: broadcast::Sender<String>,
}

impl BackendLogStream {
    pub fn new() -> Self {
        let (tx, _rx) = broadcast::channel(LOG_BROADCAST_CAP);
        Self {
            head: RwLock::new(VecDeque::with_capacity(LOG_HEAD_CAP)),
            tail: RwLock::new(VecDeque::new()),
            tx,
        }
    }

    /// Push a new log line. Broadcasts to subscribers and maintains the ring buffer.
    pub async fn push(&self, line: String) {
        // Broadcast to live subscribers (ignore errors if no subscribers).
        let _ = self.tx.send(line.clone());

        let mut head = self.head.write().await;
        let mut tail = self.tail.write().await;

        head.push_back(line.clone());
        if head.len() > LOG_HEAD_CAP {
            // Shift overflow to tail
            while head.len() > LOG_HEAD_CAP {
                if let Some(overflow) = head.pop_front() {
                    tail.push_back(overflow);
                }
            }
        }
    }

    /// Get snapshot of all lines for non-streaming consumers.
    pub async fn snapshot(&self) -> Vec<String> {
        let (head, tail) = tokio::join!(self.head.read(), self.tail.read());
        let mut result: Vec<String> = head.iter().cloned().collect();
        result.extend(tail.iter().cloned());
        result
    }

    /// Create a broadcast receiver for SSE streaming.
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.tx.subscribe()
    }
}

/// Manages log streams for all backend instances.
#[derive(Default, Clone)]
pub struct BackendLogManager {
    streams: Arc<RwLock<std::collections::HashMap<String, Arc<BackendLogStream>>>>,
}

impl BackendLogManager {
    /// Get or create a log stream for a given server name.
    pub async fn get_or_create(&self, server_name: &str) -> Arc<BackendLogStream> {
        let mut streams = self.streams.write().await;
        streams
            .entry(server_name.to_string())
            .or_insert_with(|| Arc::new(BackendLogStream::new()))
            .clone()
    }

    /// Get an existing stream (returns None if not found).
    pub async fn get(&self, server_name: &str) -> Option<Arc<BackendLogStream>> {
        self.streams.read().await.get(server_name).cloned()
    }
}
