use crate::config::Config;
use crate::db::queries::upsert_update_check;
use crate::updates::checker::UpdateChecker;
use tempfile::tempdir;

#[tokio::test]
async fn test_new_checker() {
    let checker = UpdateChecker::new();
    // Should just work
    drop(checker);
}

#[tokio::test]
async fn test_get_results() {
    let dir = tempdir().unwrap();
    let config_dir = dir.path().to_path_buf();

    let config = Config::default();
    config.save_to(&config_dir).unwrap();

    let open = crate::db::open(&config_dir).unwrap();
    upsert_update_check(
        &open.conn,
        "backend",
        "test-backend",
        Some("v1"),
        Some("v2"),
        true,
        "update_available",
        None,
        None,
        123456789,
    )
    .unwrap();

    let checker = UpdateChecker::new();
    let results = checker.get_results(&config_dir).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].item_type, "backend");
    assert_eq!(results[0].item_id, "test-backend");
    assert_eq!(results[0].current_version.as_deref(), Some("v1"));
    assert_eq!(results[0].latest_version.as_deref(), Some("v2"));
    assert!(results[0].update_available);
}

#[tokio::test]
async fn test_should_check() {
    let dir = tempdir().unwrap();
    let config_dir = dir.path().to_path_buf();

    let mut config = Config::default();
    config.general.update_check_interval = 1;
    config.save_to(&config_dir).unwrap();

    let open = crate::db::open(&config_dir).unwrap();
    // No records yet, should return true
    let checker = UpdateChecker::new();
    assert!(checker.should_check(&config_dir).await.unwrap());

    // Insert a record from 2 hours ago
    let now = chrono::Utc::now().timestamp();
    let two_hours_ago = now - 7200;

    upsert_update_check(
        &open.conn,
        "backend",
        "test",
        None,
        None,
        false,
        "unknown",
        None,
        None,
        two_hours_ago,
    )
    .unwrap();

    // Interval is 1 hour, so 2 hours ago should trigger check
    assert!(checker.should_check(&config_dir).await.unwrap());
}
