use anyhow::{Context, Result};
use koji_core::config::Config;
use koji_core::db::OpenResult;

pub(super) fn cmd_enable(_config: &Config, name: &str) -> Result<()> {
    let db_dir = koji_core::config::Config::config_dir()?;
    let OpenResult { conn, .. } = koji_core::db::open(&db_dir)?;
    let mut model_configs = koji_core::db::load_model_configs(&conn)?;

    let srv = model_configs
        .get_mut(name)
        .with_context(|| format!("Model '{}' not found", name))?;
    srv.enabled = true;

    koji_core::db::save_model_config(&conn, name, srv)?;
    println!("Enabled model: {}", name);
    Ok(())
}

pub(super) fn cmd_disable(_config: &Config, name: &str) -> Result<()> {
    let db_dir = koji_core::config::Config::config_dir()?;
    let OpenResult { conn, .. } = koji_core::db::open(&db_dir)?;
    let mut model_configs = koji_core::db::load_model_configs(&conn)?;

    let srv = model_configs
        .get_mut(name)
        .with_context(|| format!("Model '{}' not found", name))?;
    srv.enabled = false;

    koji_core::db::save_model_config(&conn, name, srv)?;
    println!("Disabled model: {}", name);
    Ok(())
}
