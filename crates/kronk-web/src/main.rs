#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let addr: std::net::SocketAddr = "0.0.0.0:11435".parse()?;
    let proxy_base_url =
        std::env::var("KRONK_PROXY_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    kronk_web::server::run(addr, proxy_base_url).await
}
