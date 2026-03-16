use anyhow::{Context, Result};
use directories::UserDirs;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub general: General,
    pub backends: Backends,
    pub profiles: Profiles,
    pub supervisor: Supervisor,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct General {
    pub log_level: String,
    pub data_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Backends {
    #[serde(default)]
    pub ik_llama: BackendConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendConfig {
    pub path: String,
    pub default_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Profiles {
    #[serde(default)]
    pub speed: ProfileConfig,
    #[serde(default)]
    pub precision: ProfileConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileConfig {
    pub backend: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Supervisor {
    pub restart_policy: String,
    pub max_restarts: u32,
    pub restart_delay_ms: u64,
    pub health_check_interval_ms: u64,
    pub hang_timeout_ms: u64,
}

impl Config {
    pub fn load() -> Result<Self> {
        let user_dirs = UserDirs::new().context("Failed to get user directories")?;
        let home = user_dirs.home_dir();
        
        let config_dir = home.join(".config/kronk");
        fs::create_dir_all(&config_dir).context("Failed to create config directory")?;
        
        let config_path = config_dir.join("config.toml");
        
        if config_path.exists() {
            let contents = fs::read_to_string(&config_path)
                .context("Failed to read config file")?;
            
            toml::from_str(&contents)
                .context("Failed to parse config file")
        } else {
            let default = Self::default();
            fs::write(&config_path, toml::to_string(&default).unwrap())
                .context("Failed to write default config")?;
            Ok(default)
        }
    }
}
