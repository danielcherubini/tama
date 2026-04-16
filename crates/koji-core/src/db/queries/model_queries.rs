//! Model-related database query functions.

use anyhow::Result;
use rusqlite::Connection;

use super::types::{DownloadLogEntry, ModelFileRecord, ModelPullRecord};

/// Insert or update the pull record for a model.
/// Uses INSERT ... ON CONFLICT(model_id) DO UPDATE.
/// Timestamp generated via SQLite's strftime('%Y-%m-%dT%H:%M:%fZ', 'now').
pub fn upsert_model_pull(
    conn: &Connection,
    model_id: i64,
    repo_id: &str,
    commit_sha: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO model_pulls (model_id, repo_id, commit_sha, pulled_at)
         VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT(model_id) DO UPDATE SET
             repo_id     = excluded.repo_id,
             commit_sha  = excluded.commit_sha,
             pulled_at   = excluded.pulled_at",
        (model_id, repo_id, commit_sha),
    )?;
    Ok(())
}

/// Get the stored pull record for a model. Returns None if never pulled.
pub fn get_model_pull(conn: &Connection, model_id: i64) -> Result<Option<ModelPullRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, model_id, repo_id, commit_sha, pulled_at
         FROM model_pulls WHERE model_id = ?1",
    )?;
    let mut rows = stmt.query_map([model_id], |row| {
        Ok(ModelPullRecord {
            id: row.get(0)?,
            model_id: row.get(1)?,
            repo_id: row.get(2)?,
            commit_sha: row.get(3)?,
            pulled_at: row.get(4)?,
        })
    })?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

/// Insert or update a file record for a downloaded GGUF.
/// Uses INSERT ... ON CONFLICT(model_id, filename) DO UPDATE.
/// Timestamp generated via SQLite's strftime('%Y-%m-%dT%H:%M:%fZ', 'now').
///
/// **Verification state preservation:** if a row already exists and the incoming
/// `lfs_oid` equals the stored one, the existing verification fields are preserved.
/// If the hash changed the verification columns are cleared so the file will be re-verified.
pub fn upsert_model_file(
    conn: &Connection,
    model_id: i64,
    repo_id: &str,
    filename: &str,
    quant: Option<&str>,
    lfs_oid: Option<&str>,
    size_bytes: Option<i64>,
) -> Result<()> {
    conn.execute(
        "INSERT INTO model_files
             (model_id, repo_id, filename, quant, lfs_oid, size_bytes, downloaded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT(model_id, filename) DO UPDATE SET
             repo_id       = excluded.repo_id,
             quant         = excluded.quant,
             lfs_oid       = excluded.lfs_oid,
             size_bytes    = excluded.size_bytes,
             downloaded_at = excluded.downloaded_at,
             -- Only clear verification when the hash actually changed.
             last_verified_at = CASE
                 WHEN model_files.lfs_oid IS NOT excluded.lfs_oid THEN NULL
                 ELSE model_files.last_verified_at END,
             verified_ok = CASE
                 WHEN model_files.lfs_oid IS NOT excluded.lfs_oid THEN NULL
                 ELSE model_files.verified_ok END,
             verify_error = CASE
                 WHEN model_files.lfs_oid IS NOT excluded.lfs_oid THEN NULL
                 ELSE model_files.verify_error END",
        (model_id, repo_id, filename, quant, lfs_oid, size_bytes),
    )?;
    Ok(())
}

/// Update the verification columns for a single file.
///
/// - `verified_ok = Some(true)`: hash matched; `verify_error` cleared.
/// - `verified_ok = Some(false)`: hash mismatch; caller should supply a short `verify_error`.
/// - `verified_ok = None`: no upstream hash available.
///
/// `last_verified_at` is set to the current time via SQLite's `strftime`.
pub fn update_verification(
    conn: &Connection,
    model_id: i64,
    filename: &str,
    verified_ok: Option<bool>,
    verify_error: Option<&str>,
) -> Result<()> {
    let verified_ok_int = verified_ok.map(|b| b as i64);
    let verify_error_param = if verified_ok == Some(true) {
        None
    } else {
        verify_error
    };
    let affected = conn.execute(
        "UPDATE model_files SET
              last_verified_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
              verified_ok      = ?3,
              verify_error     = ?4
          WHERE model_id = ?1 AND filename = ?2",
        (model_id, filename, verified_ok_int, verify_error_param),
    )?;
    if affected == 0 {
        anyhow::bail!(
            "update_verification: no row found for model_id={} filename={}",
            model_id,
            filename
        );
    }
    Ok(())
}

/// Get all stored file records for a model.
pub fn get_model_files(conn: &Connection, model_id: i64) -> Result<Vec<ModelFileRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, model_id, repo_id, filename, quant, lfs_oid, size_bytes, downloaded_at,
                last_verified_at, verified_ok, verify_error
         FROM model_files WHERE model_id = ?1",
    )?;
    let rows = stmt.query_map([model_id], |row| {
        let verified_ok: Option<i64> = row.get(9)?;
        Ok(ModelFileRecord {
            id: row.get(0)?,
            model_id: row.get(1)?,
            repo_id: row.get(2)?,
            filename: row.get(3)?,
            quant: row.get(4)?,
            lfs_oid: row.get(5)?,
            size_bytes: row.get(6)?,
            downloaded_at: row.get(7)?,
            last_verified_at: row.get(8)?,
            verified_ok: verified_ok.map(|v| v != 0),
            verify_error: row.get(10)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Get all stored file records across all models.
pub fn get_all_model_files(conn: &Connection) -> Result<Vec<ModelFileRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, model_id, repo_id, filename, quant, lfs_oid, size_bytes, downloaded_at,
                last_verified_at, verified_ok, verify_error
         FROM model_files",
    )?;
    let rows = stmt.query_map([], |row| {
        let verified_ok: Option<i64> = row.get(9)?;
        Ok(ModelFileRecord {
            id: row.get(0)?,
            model_id: row.get(1)?,
            repo_id: row.get(2)?,
            filename: row.get(3)?,
            quant: row.get(4)?,
            lfs_oid: row.get(5)?,
            size_bytes: row.get(6)?,
            downloaded_at: row.get(7)?,
            last_verified_at: row.get(8)?,
            verified_ok: verified_ok.map(|v| v != 0),
            verify_error: row.get(10)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Delete all records for a model (model_pulls, model_files cascade automatically).
/// Does NOT delete download_log entries (they're historical).
pub fn delete_model_records(conn: &Connection, model_id: i64) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute("DELETE FROM model_configs WHERE id = ?1", [model_id])?;
    // model_pulls and model_files are deleted by ON DELETE CASCADE
    tx.commit()?;
    Ok(())
}

/// Delete a single model file record by (model_id, filename).
/// Does NOT touch model_pulls — the pull record stays.
pub fn delete_model_file(conn: &Connection, model_id: i64, filename: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM model_files WHERE model_id = ?1 AND filename = ?2",
        (model_id, filename),
    )?;
    Ok(())
}

/// Log a download event (append-only).
pub fn log_download(conn: &Connection, entry: &DownloadLogEntry) -> Result<()> {
    conn.execute(
        "INSERT INTO download_log
             (repo_id, filename, started_at, completed_at,
              size_bytes, duration_ms, success, error_message)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        (
            &entry.repo_id,
            &entry.filename,
            &entry.started_at,
            entry.completed_at.as_deref(),
            entry.size_bytes,
            entry.duration_ms,
            entry.success as i64,
            entry.error_message.as_deref(),
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::db::OpenResult;

    #[test]
    fn test_delete_model_file() {
        let OpenResult { conn, .. } = db::open_in_memory().unwrap();

        // Insert a model config first (required for FK)
        db::save_model_config(&conn, "test-repo", &Default::default()).unwrap();

        // Insert a model file record via upsert
        upsert_model_file(
            &conn,
            1,
            "test-repo",
            "test-model.Q4_K_M.gguf",
            Some("Q4_K_M"),
            Some("sha256-abc123"),
            Some(1_000_000),
        )
        .unwrap();

        // Verify it exists
        let files_before = get_model_files(&conn, 1).unwrap();
        assert_eq!(files_before.len(), 1);
        assert_eq!(files_before[0].filename, "test-model.Q4_K_M.gguf");

        // Delete the file record
        delete_model_file(&conn, 1, "test-model.Q4_K_M.gguf").unwrap();

        // Verify it's gone
        let files_after = get_model_files(&conn, 1).unwrap();
        assert_eq!(files_after.len(), 0);
    }

    #[test]
    fn test_delete_model_file_preserves_other_files() {
        let OpenResult { conn, .. } = db::open_in_memory().unwrap();

        db::save_model_config(&conn, "test-repo", &Default::default()).unwrap();

        upsert_model_file(
            &conn,
            1,
            "test-repo",
            "model.Q4_K_M.gguf",
            Some("Q4_K_M"),
            None,
            None,
        )
        .unwrap();
        upsert_model_file(
            &conn,
            1,
            "test-repo",
            "model.Q8_0.gguf",
            Some("Q8_0"),
            None,
            None,
        )
        .unwrap();

        delete_model_file(&conn, 1, "model.Q4_K_M.gguf").unwrap();

        let files = get_model_files(&conn, 1).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filename, "model.Q8_0.gguf");
    }

    #[test]
    fn test_delete_model_file_nonexistent() {
        let OpenResult { conn, .. } = db::open_in_memory().unwrap();
        db::save_model_config(&conn, "test-repo", &Default::default()).unwrap();

        // Should succeed (no-op)
        delete_model_file(&conn, 1, "nonexistent.gguf").unwrap();
        let files = get_model_files(&conn, 1).unwrap();
        assert_eq!(files.len(), 0);
    }
}
