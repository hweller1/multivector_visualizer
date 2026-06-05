use serde::Deserialize;
use crate::viz::VizRepl;

#[derive(Debug, Deserialize)]
pub struct ScenarioFile {
    pub meta:   ScenarioMeta,
    pub corpus: CorpusDef,
    pub steps:  Vec<StepDef>,
}

#[derive(Debug, Deserialize)]
pub struct ScenarioMeta {
    pub title:       String,
    pub engine:      String,
    pub description: String,
    pub version:     u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CorpusDef {
    /// TOML: [corpus]\n type = "shared"
    Shared,
    /// TOML: [corpus]\n type = "inline"\n [[corpus.docs]]
    Inline { docs: Vec<InlineDoc> },
    /// TOML: [corpus]\n type = "file"\n path = "..."
    File   { path: String },
}

#[derive(Debug, Deserialize)]
pub struct InlineDoc {
    pub id:   u32,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct StepDef {
    pub op:        String,
    pub args:      Vec<String>,
    pub narration: String,
    #[serde(default)]
    pub pause:     bool,
}

pub struct ScenarioRunner {
    pub scenario: ScenarioFile,
    pub dry_run:  bool,
}

impl ScenarioRunner {
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let scenario: ScenarioFile = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("scenario parse error: {e}"))?;
        if scenario.meta.version != 1 {
            anyhow::bail!("unsupported scenario version {}", scenario.meta.version);
        }
        Ok(Self { scenario, dry_run: false })
    }

    /// Execute steps sequentially. Calls engine dispatch fn for each op.
    pub async fn run<F, Fut>(&self, mut dispatch: F) -> anyhow::Result<()>
    where
        F: FnMut(String, Vec<String>) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
    {
        for step in &self.scenario.steps {
            VizRepl::print_narration(&step.narration);
            if self.dry_run {
                continue;
            }
            dispatch(step.op.clone(), step.args.clone()).await?;
        }
        Ok(())
    }
}
