use anyhow::Result;
use clap::Parser;
use kronk_core::config::Config;

#[derive(Parser, Debug)]
#[command(name = "kronk")]
#[command(author = "Kronk Team")]
#[command(version = "0.1.0")]
#[command(about = "The Heavy-Lifting Henchman for Local AI", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Parser, Debug)]
enum Commands {
    Run { profile: String },
    Pull { model: String },
    Service {
        #[command(subcommand)]
        command: ServiceCommands,
    },
    Status,
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

#[derive(Parser, Debug)]
enum ServiceCommands {
    Install { profile: String },
    Remove { profile: String },
    Start { profile: String },
    Stop { profile: String },
}

#[derive(Parser, Debug)]
enum ConfigCommands {
    Show,
    Edit,
}

fn main() -> Result<()> {
    let _args = Args::parse();

    let config = Config::load()?;
    println!("KRONK: The Heavy-Lifting Henchman for Local AI");
    println!("Config loaded from: {}", config.general.data_dir);

    Ok(())
}
