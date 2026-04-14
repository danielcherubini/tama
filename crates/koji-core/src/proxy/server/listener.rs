use axum::Router;
use std::net::SocketAddr;
use tracing::info;

/// Start the proxy server on the given address.
///
/// Binds a TCP listener and serves the provided router until shutdown.
/// Handles SIGTERM/SIGINT for graceful shutdown.
pub async fn run(app: Router, addr: SocketAddr) -> anyhow::Result<()> {
    info!("Starting proxy server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Create a future that completes when we receive SIGTERM or SIGINT
    let shutdown_signal = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};

            let mut sigterm =
                signal(SignalKind::terminate()).expect("Failed to install SIGTERM handler");
            let mut sigint =
                signal(SignalKind::interrupt()).expect("Failed to install SIGINT handler");

            tokio::select! {
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, shutting down...");
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT, shutting down...");
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("Received Ctrl+C, shutting down...");
                }
            }
        }

        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.ok();
            info!("Received Ctrl+C, shutting down...");
        }
    };

    // Run the server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("Server shutdown complete");
    Ok(())
}
