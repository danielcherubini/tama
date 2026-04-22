use anyhow::Result;
use rusqlite::Connection;

use super::types::UpdateCheckRecord;

/// Params for upserting an update check record.
#[derive(Debug, Clone)]
pub struct UpdateCheckParams<'a> {
    pub item_type: &'a str,
    pub item_id: &'a str,
    pub current_version: Option<&'a str>,
    pub latest_version: Option<&'a str>,
    pub update_available: bool,
    pub status: &'a str,
    pub error_message: Option<&'a str>,
    pub details_json: Option<&'a str>,
    pub checked_at: i64,
}

pub fn upsert_update_check(conn: &Connection, params: UpdateCheckParams) -> Result<()> {
    conn.execute(
        "INSERT INTO update_checks (item_type, item_id, current_version, latest_version, update_available, status, error_message, details_json, checked_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(item_type, item_id) DO UPDATE SET
             current_version = excluded.current_version,
             latest_version = excluded.latest_version,
             update_available = excluded.update_available,
             status = excluded.status,
             error_message = excluded.error_message,
             details_json = excluded.details_json,
             checked_at = excluded.checked_at",
        (
            params.item_type,
            params.item_id,
            params.current_version,
            params.latest_version,
            params.update_available as i32,
            params.status,
            params.error_message,
            params.details_json,
            params.checked_at,
        ),
    )?;
    Ok(())
}

pub fn get_all_update_checks(conn: &Connection) -> Result<Vec<UpdateCheckRecord>> {
    let mut stmt = conn.prepare(
        "SELECT item_type, item_id, current_version, latest_version, update_available, status, error_message, details_json, checked_at
         FROM update_checks ORDER BY item_type, item_id",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(UpdateCheckRecord {
            item_type: row.get(0)?,
            item_id: row.get(1)?,
            current_version: row.get(2)?,
            latest_version: row.get(3)?,
            update_available: row.get::<_, i32>(4)? != 0,
            status: row.get(5)?,
            error_message: row.get(6)?,
            details_json: row.get(7)?,
            checked_at: row.get(8)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

pub fn get_update_check(
    conn: &Connection,
    item_type: &str,
    item_id: &str,
) -> Result<Option<UpdateCheckRecord>> {
    let mut stmt = conn.prepare(
        "SELECT item_type, item_id, current_version, latest_version, update_available, status, error_message, details_json, checked_at
         FROM update_checks WHERE item_type = ?1 AND item_id = ?2",
    )?;
    let mut rows = stmt.query_map((item_type, item_id), |row| {
        Ok(UpdateCheckRecord {
            item_type: row.get(0)?,
            item_id: row.get(1)?,
            current_version: row.get(2)?,
            latest_version: row.get(3)?,
            update_available: row.get::<_, i32>(4)? != 0,
            status: row.get(5)?,
            error_message: row.get(6)?,
            details_json: row.get(7)?,
            checked_at: row.get(8)?,
        })
    })?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

pub fn delete_update_check(conn: &Connection, item_type: &str, item_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM update_checks WHERE item_type = ?1 AND item_id = ?2",
        (item_type, item_id),
    )?;
    Ok(())
}

pub fn get_oldest_check_time(conn: &Connection) -> Result<Option<i64>> {
    let mut stmt = conn.prepare("SELECT MIN(checked_at) FROM update_checks")?;
    let mut rows = stmt.query_map([], |row| row.get::<_, Option<i64>>(0))?;
    match rows.next() {
        Some(row) => Ok(row?),
        None => Ok(None),
    }
}
