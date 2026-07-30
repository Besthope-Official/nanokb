use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    nanokb::cli::run().await
}