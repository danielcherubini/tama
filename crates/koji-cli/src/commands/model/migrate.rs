use anyhow::Result;
use koji_core::config::Config;
use koji_core::db::OpenResult;

pub(super) fn cmd_migrate(config: &Config) -> Result<()> {
    let db_dir = koji_core::config::Config::config_dir()?;
    let OpenResult { conn, .. } = koji_core::db::open(&db_dir)?;

    // We need a mutable config to call migrate_models_to_db.
    let mut mutable_config = config.clone();

    let migrated =
        koji_core::config::migrate::model_to_db::migrate_models_to_db(&conn, &mut mutable_config)?;

    if migrated == 0 {
        println!("Nothing to migrate.");
    } else {
        println!("Successfully migrated {} models to the database.", migrated);
    }

    Ok(())
}
