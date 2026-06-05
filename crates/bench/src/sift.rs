use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

pub struct SiftDownloader {
    pub data_dir: PathBuf,
}

impl SiftDownloader {
    pub fn new() -> Self {
        let data_dir = std::env::var("MULTIVECTOR_SIFT_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/sift"));
        Self { data_dir }
    }

    pub fn ensure_present(&self) -> Result<PathBuf> {
        let base_path = self.data_dir.join("bigann_base.bvecs");
        let gnd_path = self.data_dir.join("bigann_gnd/idx_100M.ivecs");

        if base_path.exists() && gnd_path.exists() {
            return Ok(self.data_dir.clone());
        }

        eprintln!("SIFT data not found at {:?}", self.data_dir);
        eprintln!("To download SIFT-1B:");
        eprintln!("  mkdir -p {:?}", self.data_dir);
        eprintln!("  ftp ftp://ftp.irisa.fr/local/texmex/corpus/");
        eprintln!("  get bigann_base.bvecs");
        eprintln!("  get bigann_gnd.tar.gz");
        eprintln!("  tar xf bigann_gnd.tar.gz");
        eprintln!("Or set MULTIVECTOR_SIFT_PATH to the directory containing bigann_base.bvecs");

        Err(anyhow!(
            "SIFT data missing at {:?}. See above for download instructions.",
            self.data_dir
        ))
    }

    pub fn read_bvecs(&self, path: &Path) -> Result<Vec<Vec<f32>>> {
        let data = std::fs::read(path)?;
        let mut offset = 0usize;
        let mut vecs = Vec::new();
        while offset + 4 <= data.len() {
            let dim = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + dim > data.len() {
                break;
            }
            let v: Vec<f32> = data[offset..offset + dim]
                .iter()
                .map(|&b| b as f32)
                .collect();
            vecs.push(v);
            offset += dim;
        }
        Ok(vecs)
    }

    pub fn read_ivecs(&self, path: &Path) -> Result<Vec<Vec<i32>>> {
        let data = std::fs::read(path)?;
        let mut offset = 0usize;
        let mut vecs = Vec::new();
        while offset + 4 <= data.len() {
            let dim = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + dim * 4 > data.len() {
                break;
            }
            let v: Vec<i32> = (0..dim)
                .map(|i| {
                    i32::from_le_bytes([
                        data[offset + i * 4],
                        data[offset + i * 4 + 1],
                        data[offset + i * 4 + 2],
                        data[offset + i * 4 + 3],
                    ])
                })
                .collect();
            vecs.push(v);
            offset += dim * 4;
        }
        Ok(vecs)
    }
}

impl Default for SiftDownloader {
    fn default() -> Self {
        Self::new()
    }
}
