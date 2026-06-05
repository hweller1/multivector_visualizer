use super::python_engine::{ColbertPython, PythonEngine, WarpPython};
use super::sift::SiftDownloader;
use common::bench_types::{BenchResult, PlotData};

pub struct BenchRunner;

impl BenchRunner {
    pub fn new() -> Self {
        Self
    }

    pub async fn run_all(&self) -> anyhow::Result<PlotData> {
        let mut all_results: Vec<BenchResult> = Vec::new();

        let engines: Vec<Box<dyn PythonEngine>> =
            vec![Box::new(ColbertPython), Box::new(WarpPython)];

        for engine in &engines {
            if !engine.check_installed() {
                println!(
                    "Engine '{}' not installed. Install from: {}",
                    engine.name(),
                    engine.install_url()
                );
                continue;
            }

            println!("Running build_index for '{}'...", engine.name());
            let build_results = engine.build_index().await?;
            all_results.extend(build_results);

            println!("Running search for '{}'...", engine.name());
            let search_results = engine.search().await?;
            all_results.extend(search_results);
        }

        // Write output JSON
        std::fs::create_dir_all("output")?;
        let json = serde_json::to_string_pretty(&all_results)?;
        std::fs::write("output/bench_results.json", &json)?;
        println!("Results written to output/bench_results.json");

        Ok(PlotData {
            results: all_results,
        })
    }

    pub async fn check_sift(&self) -> anyhow::Result<()> {
        let downloader = SiftDownloader::new();
        match downloader.ensure_present() {
            Ok(path) => println!("SIFT data found at {:?}", path),
            Err(e) => println!("SIFT check failed: {e}"),
        }
        Ok(())
    }
}

impl Default for BenchRunner {
    fn default() -> Self {
        Self::new()
    }
}
