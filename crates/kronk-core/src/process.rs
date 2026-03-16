use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use std::process::Stdio;

use crate::config::{BackendConfig, Config};

#[derive(Debug, Clone)]
pub enum ProcessEvent {
    Started,
    Output(String),
    Crashed,
    Restarting,
    Stopped,
    HealthSnapshot(HealthSnapshot),
}

#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    pub alive: bool,
    pub restart_count: u32,
    pub uptime_ms: u64,
}

pub struct ProcessSupervisor {
    config: Config,
    backend: BackendConfig,
    restart_count: u32,
    last_output_time: Option<std::time::Instant>,
    hang_timeout_ms: u64,
}

impl ProcessSupervisor {
    pub fn new(config: Config, backend: BackendConfig, hang_timeout_ms: u64) -> Self {
        Self {
            config,
            backend,
            restart_count: 0,
            last_output_time: None,
            hang_timeout_ms,
        }
    }

    pub async fn run(self) -> Result<()> {
        let (tx, mut rx) = mpsc::unbounded_channel::<ProcessEvent>();
        
        let mut supervisor = self;
        
        // Spawn child process
        let mut child = Command::new(&supervisor.backend.path)
            .args(&supervisor.backend.default_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn backend process")?;

        tx.send(ProcessEvent::Started).ok();

        let mut health_interval = interval(Duration::from_millis(supervisor.config.supervisor.health_check_interval_ms));

        loop {
            tokio::select! {
                Some(event) = rx.recv() => {
                    match event {
                        ProcessEvent::Started => {}
                        ProcessEvent::Output(line) => {
                            tracing::debug!("Backend output: {}", line);
                        }
                        ProcessEvent::Crashed => {
                            supervisor.handle_crash(tx.clone()).await?;
                        }
                        ProcessEvent::Stopped => {
                            tracing::info!("Backend stopped");
                            return Ok(());
                        }
                        ProcessEvent::HealthSnapshot(snapshot) => {
                            tracing::debug!("Health snapshot: {:?}", snapshot);
                        }
                        _ => {}
                    }
                }
                _ = health_interval.tick() => {
                    supervisor.health_check(&mut child, &tx).await?;
                }
            }
        }
    }

    async fn handle_crash(&mut self, tx: mpsc::UnboundedSender<ProcessEvent>) -> Result<()> {
        self.restart_count += 1;
        tx.send(ProcessEvent::Crashed).ok();
        tx.send(ProcessEvent::Restarting).ok();

        let delay = Duration::from_millis(self.config.supervisor.restart_delay_ms);
        tokio::time::sleep(delay).await;

        let child = Command::new(&self.backend.path)
            .args(&self.backend.default_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to restart backend")?;

        tx.send(ProcessEvent::Started).ok();
        Ok(())
    }

    async fn health_check(&mut self, child: &mut tokio::process::Child, tx: &mpsc::UnboundedSender<ProcessEvent>) -> Result<()> {
        let alive = !child.try_wait().unwrap().is_some();
        self.last_output_time = Some(std::time::Instant::now());

        if !alive {
            tx.send(ProcessEvent::Stopped).ok();
            return Ok(());
        }

        tx.send(ProcessEvent::HealthSnapshot(HealthSnapshot {
            alive,
            restart_count: self.restart_count,
            uptime_ms: 0,
        })).ok();

        Ok(())
    }
}
