use anyhow::{Context, Result};

#[cfg(feature = "windows-service")]
use windows_service::{
    service_type::ServiceType,
    status::ServiceStatus,
    server::{Service, ServiceRunner},
};

#[cfg(feature = "windows-service")]
pub struct ServiceManager {
    service_name: String,
}

#[cfg(feature = "windows-service")]
impl ServiceManager {
    pub fn new(name: &str) -> Self {
        Self {
            service_name: name.to_string(),
        }
    }

    pub fn install(&self) -> Result<()> {
        Ok(())
    }

    pub fn remove(&self) -> Result<()> {
        Ok(())
    }

    pub fn start(&self) -> Result<()> {
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        Ok(())
    }

    pub fn status(&self) -> Result<String> {
        Ok("running".to_string())
    }
}

#[cfg(not(feature = "windows-service"))]
pub struct ServiceManager;

#[cfg(not(feature = "windows-service"))]
impl ServiceManager {
    pub fn new(_name: &str) -> Self {
        Self
    }

    pub fn install(&self) -> Result<()> {
        Ok(())
    }

    pub fn remove(&self) -> Result<()> {
        Ok(())
    }

    pub fn start(&self) -> Result<()> {
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        Ok(())
    }

    pub fn status(&self) -> Result<String> {
        Ok("disabled".to_string())
    }
}
