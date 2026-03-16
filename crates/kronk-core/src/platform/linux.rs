use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub struct ServiceManager {
    config_dir: PathBuf,
}

impl ServiceManager {
    pub fn new() -> Self {
        let home = dirs::home_dir().expect("Home directory not found");
        Self {
            config_dir: home.join(".config/systemd/user"),
        }
    }

    pub fn install(&self, name: &str) -> Result<()> {
        fs::create_dir_all(&self.config_dir).context("Failed to create config dir")?;
        let unit_path = self.config_dir.join(format!("{}.service", name));
        
        let unit = format!(
            "[Unit]
Description={}
After=network.target

[Service]
Type=simple
ExecStart={}
Restart=always
RestartSec={}

[Install]
WantedBy=default.target
",
            name,
            std::env::current_dir()?.display(),
            "1000"
        );
        
        fs::write(&unit_path, unit).context("Failed to write unit file")?;
        
        Ok(())
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        let unit_path = self.config_dir.join(format!("{}.service", name));
        if unit_path.exists() {
            fs::remove_file(&unit_path).context("Failed to remove unit file")?;
        }
        Ok(())
    }

    pub fn start(&self, name: &str) -> Result<()> {
        Ok(())
    }

    pub fn stop(&self, name: &str) -> Result<()> {
        Ok(())
    }

    pub fn status(&self, name: &str) -> Result<String> {
        Ok("unknown".to_string())
    }
}
