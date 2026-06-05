use async_trait::async_trait;
use common::bench_types::BenchResult;
use tokio::process::Command;
use which::which;

#[async_trait]
pub trait PythonEngine: Send + Sync {
    fn name(&self) -> &str;
    fn binary_name(&self) -> &str;
    fn check_installed(&self) -> bool {
        which(self.binary_name()).is_ok()
    }
    fn install_url(&self) -> &str;
    async fn build_index(&self) -> anyhow::Result<Vec<BenchResult>>;
    async fn search(&self) -> anyhow::Result<Vec<BenchResult>>;
    fn parse_line(&self, line: &str) -> Option<BenchResult> {
        serde_json::from_str(line).ok()
    }
}

pub struct ColbertPython;
pub struct WarpPython;

#[async_trait]
impl PythonEngine for ColbertPython {
    fn name(&self) -> &str {
        "colbert-python"
    }

    fn binary_name(&self) -> &str {
        "python3"
    }

    fn install_url(&self) -> &str {
        "https://github.com/stanford-futuredata/ColBERT"
    }

    async fn build_index(&self) -> anyhow::Result<Vec<BenchResult>> {
        let output = Command::new("python3")
            .args(["-c", "print('colbert build_index: not implemented')"])
            .output()
            .await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .filter_map(|line| self.parse_line(line))
            .collect())
    }

    async fn search(&self) -> anyhow::Result<Vec<BenchResult>> {
        let output = Command::new("python3")
            .args(["-c", "print('colbert search: not implemented')"])
            .output()
            .await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .filter_map(|line| self.parse_line(line))
            .collect())
    }
}

#[async_trait]
impl PythonEngine for WarpPython {
    fn name(&self) -> &str {
        "warp-python"
    }

    fn binary_name(&self) -> &str {
        "python3"
    }

    fn install_url(&self) -> &str {
        "https://github.com/stanford-futuredata/WARP"
    }

    async fn build_index(&self) -> anyhow::Result<Vec<BenchResult>> {
        let output = Command::new("python3")
            .args(["-c", "print('warp build_index: not implemented')"])
            .output()
            .await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .filter_map(|line| self.parse_line(line))
            .collect())
    }

    async fn search(&self) -> anyhow::Result<Vec<BenchResult>> {
        let output = Command::new("python3")
            .args(["-c", "print('warp search: not implemented')"])
            .output()
            .await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .filter_map(|line| self.parse_line(line))
            .collect())
    }
}
