use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    nanokb::eval::run().await
}
