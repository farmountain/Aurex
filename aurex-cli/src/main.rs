use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about = "AUREX command line interface")]
struct Cli {
    /// Backend target to use (e.g., cpu, rocm, vulkan)
    #[arg(long, default_value = "cpu")]
    target: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a model for the selected backend
    Compile { model: String },
    /// Run inference using a compiled model
    Run { model: String },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { model } => {
            aurex_cli::compile_model(&model, &cli.target);
        }
        Commands::Run { model } => {
            aurex_cli::run_model(&model, &cli.target);
        }
    }
}
