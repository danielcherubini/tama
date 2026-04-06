#[cfg(feature = "ssr")]
mod tests {
    use std::sync::Arc;

    async fn start_test_server() -> (reqwest::Client, std::net::SocketAddr) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let state = Arc::new(kronk_web::server::AppState {
                proxy_base_url: "http://127.0.0.1:11434".to_string(),
                client: reqwest::Client::new(),
                logs_dir: None,
                config_path: None,
                proxy_config: None,
            });
            axum::serve(listener, kronk_web::server::build_router(state))
                .await
                .unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (reqwest::Client::new(), addr)
    }

    /// GET / returns 200 (index.html embedded) or 404 (dist/ empty in dev) — both are valid.
    #[tokio::test]
    async fn test_root_returns_html_or_not_found() {
        let (client, addr) = start_test_server().await;
        let resp = client
            .get(format!("http://{}/", addr))
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        assert!(
            status == 200 || status == 404,
            "Expected 200 or 404 for /, got {status}"
        );
    }

    /// GET /api/config returns 404 when config_path is None (not configured).
    #[tokio::test]
    async fn test_api_config_returns_404_when_unconfigured() {
        let (client, addr) = start_test_server().await;
        let resp = client
            .get(format!("http://{}/api/config", addr))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body.get("error").is_some(),
            "Expected error field in response"
        );
    }

    /// GET /api/logs returns 404 when logs_dir is None (not configured).
    #[tokio::test]
    async fn test_api_logs_returns_404_when_unconfigured() {
        let (client, addr) = start_test_server().await;
        let resp = client
            .get(format!("http://{}/api/logs", addr))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 404);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(
            body.get("error").is_some(),
            "Expected error field in response"
        );
    }

    /// POST /api/config returns 404 when config_path is None (checked before TOML validation).
    #[tokio::test]
    async fn test_api_config_save_returns_404_when_unconfigured() {
        let (client, addr) = start_test_server().await;
        let resp = client
            .post(format!("http://{}/api/config", addr))
            .json(&serde_json::json!({ "content": "not valid toml [[[[" }))
            .send()
            .await
            .unwrap();
        // 404 because config_path is None (checked before TOML validation)
        assert_eq!(resp.status().as_u16(), 404);
    }
}
