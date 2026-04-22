use anyhow::{Context, Result};
use tama_core::config::Config;
use tama_core::db::OpenResult;

pub(super) fn cmd_enable(_config: &Config, name: &str) -> Result<()> {
    let db_dir = tama_core::config::Config::config_dir()?;
    let OpenResult { conn, .. } = tama_core::db::open(&db_dir)?;
    let mut model_configs = tama_core::db::load_model_configs(&conn)?;

    let srv = model_configs
        .get_mut(name)
        .with_context(|| format!("Model '{}' not found", name))?;
    srv.enabled = true;

    tama_core::db::save_model_config(&conn, name, srv)?;
    println!("Enabled model: {}", name);
    Ok(())
}

pub(super) fn cmd_disable(_config: &Config, name: &str) -> Result<()> {
    let db_dir = tama_core::config::Config::config_dir()?;
    let OpenResult { conn, .. } = tama_core::db::open(&db_dir)?;
    let mut model_configs = tama_core::db::load_model_configs(&conn)?;

    let srv = model_configs
        .get_mut(name)
        .with_context(|| format!("Model '{}' not found", name))?;
    srv.enabled = false;

    tama_core::db::save_model_config(&conn, name, srv)?;
    println!("Disabled model: {}", name);
    Ok(())
}
