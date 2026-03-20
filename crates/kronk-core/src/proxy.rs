pub mod server;

use crate::backends::registry::BackendRegistry;
use crate::config::ProxyConfig;
use crate::models::card::ModelCard;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Represents the state of a loaded model.
#[derive(Debug, Clone)]
pub struct ModelState {
    pub model_name: String,
    pub backend: String,
    pub backend_pid: Option<u32>,
    pub backend_url: String,
    pub load_time: Instant,
    pub last_accessed: Instant,
}

/// Manages proxy state and model lifecycle.
pub struct ProxyState {
    pub config: ProxyConfig,
    pub models: Arc<RwLock<HashMap<String, ModelState>>>,
    pub registry: Arc<RwLock<BackendRegistry>>,
    pub config_data: Arc<RwLock<crate::config::Config>>,
    pub process_map: Arc<Mutex<HashMap<u32, String>>>,
}

impl ProxyState {
    pub fn new(
        config: ProxyConfig,
        registry: BackendRegistry,
        config_data: crate::config::Config,
    ) -> Self {
        Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(RwLock::new(registry)),
            config_data: Arc::new(RwLock::new(config_data)),
            process_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get the backend URL for a server name.
    pub async fn get_backend_url(&self, server_name: &str) -> Result<String> {
        let config = self.config_data.read().await;
        let server = config
            .servers
            .get(server_name)
            .with_context(|| format!("Server '{}' not found", server_name))?;

        let backend_url = config
            .resolve_health_url(server)
            .with_context(|| format!("No backend URL resolved for server '{}'", server_name))?;

        Ok(backend_url)
    }

    /// Check if a model is already loaded.
    pub async fn is_model_loaded(&self, model_name: &str) -> bool {
        let models = self.models.read().await;
        models.contains_key(model_name)
    }

    /// Get the state of a loaded model.
    pub async fn get_model_state(&self, model_name: &str) -> Option<ModelState> {
        let models = self.models.read().await;
        models.get(model_name).cloned()
    }

    /// Load a model by starting its backend process.
    pub async fn load_model(
        &self,
        model_name: &str,
        _model_card: Option<&ModelCard>,
    ) -> Result<()> {
        debug!("Loading model: {}", model_name);

        // Get backend config from registry
        let backend_info = self
            .registry
            .read()
            .await
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("No backend configured for model: {}", model_name))?
            .clone();

        let backend_name = backend_info.name.clone();
        let backend_path = backend_info.path.to_string_lossy().to_string();

        // Get args from config
        let config = self.config_data.read().await;
        let (server, backend_config) = config
            .resolve_server(model_name)
            .with_context(|| format!("No server configured for model: {}", model_name))?;

        let args = config.build_args(server, backend_config);
        let health_url = config
            .resolve_health_url(server)
            .with_context(|| format!("No health URL resolved for model: {}", model_name))?;

        drop(config);

        info!(
            "Starting backend '{}' for model '{}'",
            backend_name, model_name
        );

        let start = Instant::now();
        let mut child = std::process::Command::new(&backend_path)
            .args(&args)
            .env("MODEL_NAME", model_name)
            .spawn()
            .with_context(|| format!("Failed to start backend '{}'", backend_name))?;

        let pid = child.id();
        info!(
            "Backend '{}' started for model '{}' (pid: {:?})",
            backend_name, model_name, pid
        );

        // Register PID in process map
        let mut processes = self.process_map.lock().await;
        processes.insert(pid, model_name.to_string());

        // Wait for health check to pass
        let timeout = Duration::from_secs(30);
        let mut elapsed = Duration::from_secs(0);

        while elapsed < timeout {
            tokio::time::sleep(Duration::from_millis(500)).await;
            elapsed += Duration::from_millis(500);

            if let Ok(response) = check_health(&health_url).await {
                if response.status().is_success() {
                    debug!("Health check passed for model: {}", model_name);
                    break;
                }
            }
        }

        if elapsed >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = processes.remove(&pid);
            return Err(anyhow::anyhow!(
                "Backend '{}' failed to start for model '{}' (timeout after {}s)",
                backend_name,
                model_name,
                elapsed.as_secs()
            ));
        }

        // Register the loaded model
        {
            let mut models = self.models.write().await;
            models.insert(
                model_name.to_string(),
                ModelState {
                    model_name: model_name.to_string(),
                    backend: backend_name,
                    backend_pid: Some(pid),
                    backend_url: health_url,
                    load_time: start,
                    last_accessed: start,
                },
            );
        }

        info!("Model '{}' loaded successfully", model_name);
        Ok(())
    }

    /// Unload a model by stopping its backend process.
    pub async fn unload_model(&self, model_name: &str) -> Result<()> {
        debug!("Unloading model: {}", model_name);

        let state = self
            .get_model_state(model_name)
            .await
            .with_context(|| format!("Model '{}' not loaded", model_name))?;

        let backend_name = state.backend.clone();
        let pid = state.backend_pid;

        info!(
            "Stopping backend '{}' for model '{}'",
            backend_name, model_name
        );

        // Kill the process if we have the PID
        if let Some(pid) = pid {
            info!("Sending SIGTERM to backend process {}", pid);
            // Send SIGTERM signal
            let _ = std::process::Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .spawn();

            // Wait up to 5 seconds for graceful shutdown
            tokio::time::sleep(Duration::from_secs(5)).await;
        }

        // Remove from process map
        let mut processes = self.process_map.lock().await;
        if let Some(pid) = pid {
            processes.remove(&pid);
        }

        // Remove from models
        let mut models = self.models.write().await;
        models.remove(model_name);

        info!("Model '{}' unloaded", model_name);
        Ok(())
    }

    /// Check if any model has been idle for longer than the timeout.
    pub async fn check_idle_timeouts(&self) -> Vec<String> {
        let now = Instant::now();
        let mut to_unload = Vec::new();

        let models = self.models.read().await;
        for (model_name, state) in models.iter() {
            let idle_duration = now - state.last_accessed;
            let timeout = Duration::from_secs(self.config.idle_timeout_secs);

            if idle_duration > timeout {
                warn!(
                    "Model '{}' has been idle for {}s (timeout: {}s)",
                    model_name,
                    idle_duration.as_secs(),
                    self.config.idle_timeout_secs
                );
                to_unload.push(model_name.clone());
            }
        }

        drop(models);

        // Actually unload the models
        for model_name in to_unload {
            let _ = self.unload_model(&model_name).await;
        }

        Vec::new()
    }

    /// Update the last accessed time for a model.
    pub async fn update_last_accessed(&self, model_name: &str) {
        let mut models = self.models.write().await;
        if let Some(state) = models.get_mut(model_name) {
            state.last_accessed = Instant::now();
        }
    }

    /// Get the model card for a model name.
    pub async fn get_model_card(&self, model_name: &str) -> Option<crate::models::card::ModelCard> {
        let configs_dir = self.config_data.read().await.configs_dir().ok()?;

        // Try to find the model card file
        // Format: configs.d/<company>--<model>.toml
        let card_path = configs_dir.join(format!(
            "{}--{}.toml",
            model_name.split('/').next().unwrap_or(""),
            model_name
        ));

        if card_path.exists() {
            let content = std::fs::read_to_string(&card_path).ok()?;
            let card: crate::models::card::ModelCard = toml::from_str(&content).ok()?;
            Some(card)
        } else {
            None
        }
    }
}

/// Check the health of a backend by making a request to its health endpoint.
async fn check_health(url: &str) -> Result<reqwest::Response> {
    let client = reqwest::Client::new();
    client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to check health: {}", url))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_is_model_loaded() {
        let config = ProxyConfig::default();
        let registry = BackendRegistry::default();
        let config_data = crate::config::Config::default();
        let state = ProxyState::new(config, registry, config_data);

        assert!(!state.is_model_loaded("test-model").await);
    }

    #[tokio::test]
    async fn test_get_model_state() {
        let config = ProxyConfig::default();
        let registry = BackendRegistry::default();
        let config_data = crate::config::Config::default();
        let state = ProxyState::new(config, registry, config_data);

        assert!(state.get_model_state("test-model").await.is_none());
    }

    #[tokio::test]
    async fn test_get_model_card() {
        let config = ProxyConfig::default();
        let registry = BackendRegistry::default();
        let config_data = crate::config::Config::default();
        let state = ProxyState::new(config, registry, config_data);

        let card = state.get_model_card("test-model").await;
        assert!(card.is_none());
    }
}
