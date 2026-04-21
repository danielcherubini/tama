//! TTS configuration database query functions.

use anyhow::Result;
use rusqlite::{params, Connection};

use super::types::TtsConfigRecord;

/// Insert or update the TTS engine configuration.
/// Timestamp updated via SQLite's strftime('%Y-%m-%dT%H:%M:%fZ', 'now') on conflict.
/// Returns the config id.
pub fn upsert_tts_config(conn: &Connection, record: &TtsConfigRecord) -> Result<i64> {
    conn.execute(
        "INSERT INTO tts_configs (
            engine, default_voice, speed, format, enabled,
            created_at, updated_at
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7
        )
         ON CONFLICT(engine) DO UPDATE SET
             default_voice = excluded.default_voice,
             speed = excluded.speed,
             format = excluded.format,
             enabled = excluded.enabled,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![
            record.engine,
            record.default_voice,
            record.speed,
            record.format,
            record.enabled as i32,
            record.created_at,
            record.updated_at,
        ],
    )?;
    // Return the id (either existing or newly created)
    let id: i64 = conn.query_row(
        "SELECT id FROM tts_configs WHERE engine = ?1",
        [&record.engine],
        |row| row.get(0),
    )?;
    Ok(id)
}

/// Get the TTS configuration by engine name. Returns None if not found.
pub fn get_tts_config(conn: &Connection, engine: &str) -> Result<Option<TtsConfigRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, engine, default_voice, speed, format, enabled,
                created_at, updated_at
         FROM tts_configs WHERE engine = ?1",
    )?;
    let mut rows = stmt.query_map([engine], |row| {
        Ok(TtsConfigRecord {
            id: row.get(0)?,
            engine: row.get(1)?,
            default_voice: row.get(2)?,
            speed: row.get(3)?,
            format: row.get(4)?,
            enabled: row.get::<_, i32>(5)? != 0,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    })?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

/// Get all stored TTS engine configurations.
pub fn get_all_tts_configs(conn: &Connection) -> Result<Vec<TtsConfigRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, engine, default_voice, speed, format, enabled,
                created_at, updated_at
         FROM tts_configs ORDER BY engine ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(TtsConfigRecord {
            id: row.get(0)?,
            engine: row.get(1)?,
            default_voice: row.get(2)?,
            speed: row.get(3)?,
            format: row.get(4)?,
            enabled: row.get::<_, i32>(5)? != 0,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Delete the TTS configuration by engine name.
pub fn delete_tts_config(conn: &Connection, engine: &str) -> Result<()> {
    conn.execute("DELETE FROM tts_configs WHERE engine = ?1", [engine])?;
    Ok(())
}
