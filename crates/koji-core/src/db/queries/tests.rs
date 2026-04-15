//! Tests for database query functions.

use crate::db::{open_in_memory, OpenResult};

use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_and_get_update_check() {
        let OpenResult { conn, .. } = open_in_memory().unwrap();
        let item_type = "backend";
        let item_id = "llama-cpp";
        let now = 1713168000; // 2024-04-15

        // Insert
        upsert_update_check(
            &conn,
            item_type,
            item_id,
            Some("v1.0.0"),
            Some("v1.1.0"),
            true,
            "update_available",
            None,
            None,
            now,
        )
        .unwrap();

        let record = get_update_check(&conn, item_type, item_id)
            .unwrap()
            .unwrap();
        assert_eq!(record.item_type, item_type);
        assert_eq!(record.item_id, item_id);
        assert_eq!(record.current_version.unwrap(), "v1.0.0");
        assert_eq!(record.latest_version.unwrap(), "v1.1.0");
        assert!(record.update_available);
        assert_eq!(record.status, "update_available");
        assert_eq!(record.checked_at, now);

        // Upsert (Update)
        upsert_update_check(
            &conn,
            item_type,
            item_id,
            Some("v1.1.0"),
            Some("v1.1.0"),
            false,
            "up_to_date",
            None,
            None,
            now + 100,
        )
        .unwrap();

        let updated = get_update_check(&conn, item_type, item_id)
            .unwrap()
            .unwrap();
        assert_eq!(updated.current_version.unwrap(), "v1.1.0");
        assert!(!updated.update_available);
        assert_eq!(updated.status, "up_to_date");
        assert_eq!(updated.checked_at, now + 100);
    }

    #[test]
    fn test_get_all_update_checks() {
        let OpenResult { conn, .. } = open_in_memory().unwrap();
        let now = 1713168000;

        upsert_update_check(
            &conn, "backend", "b1", None, None, false, "unknown", None, None, now,
        )
        .unwrap();

        upsert_update_check(
            &conn, "model", "m1", None, None, false, "unknown", None, None, now,
        )
        .unwrap();

        let all = get_all_update_checks(&conn).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_delete_update_check() {
        let OpenResult { conn, .. } = open_in_memory().unwrap();
        let item_type = "backend";
        let item_id = "b1";

        upsert_update_check(
            &conn, item_type, item_id, None, None, false, "unknown", None, None, 12345,
        )
        .unwrap();

        delete_update_check(&conn, item_type, item_id).unwrap();
        let record = get_update_check(&conn, item_type, item_id).unwrap();
        assert!(record.is_none());
    }

    #[test]
    fn test_get_oldest_check_time() {
        let OpenResult { conn, .. } = open_in_memory().unwrap();

        assert_eq!(get_oldest_check_time(&conn).unwrap(), None);

        upsert_update_check(
            &conn, "backend", "b1", None, None, false, "unknown", None, None, 2000,
        )
        .unwrap();

        upsert_update_check(
            &conn, "backend", "b2", None, None, false, "unknown", None, None, 1000,
        )
        .unwrap();

        assert_eq!(get_oldest_check_time(&conn).unwrap(), Some(1000));
    }
}
