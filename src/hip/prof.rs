use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[cfg(feature = "hip-prof")]
#[derive(Default)]
pub struct HipProf {
    label: &'static str,
    enabled: bool,
    totals: BTreeMap<&'static str, Duration>,
}

#[cfg(feature = "hip-prof")]
impl HipProf {
    pub fn new(label: &'static str) -> Self {
        let enabled = std::env::var("WEB_RWKV_HIP_PROF")
            .map(|v| v != "0")
            .unwrap_or(true);
        Self {
            label,
            enabled,
            totals: BTreeMap::new(),
        }
    }

    pub fn time<F, R>(&mut self, label: &'static str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        let start = Instant::now();
        let out = f();
        let elapsed = start.elapsed();
        let entry = self
            .totals
            .entry(label)
            .or_insert_with(|| Duration::from_secs(0));
        *entry += elapsed;
        out
    }

    pub fn print(&self, context: &str) {
        if !self.enabled || self.totals.is_empty() {
            return;
        }
        eprintln!("[hip-prof] {} {}", self.label, context);
        for (label, dur) in &self.totals {
            eprintln!("[hip-prof]   {:<16} {:>8.3} ms", label, dur.as_secs_f64() * 1000.0);
        }
    }
}

#[cfg(not(feature = "hip-prof"))]
#[derive(Default)]
pub struct HipProf;

#[cfg(not(feature = "hip-prof"))]
impl HipProf {
    pub fn new(_label: &'static str) -> Self {
        Self
    }

    pub fn time<F, R>(&mut self, _label: &'static str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }

    pub fn print(&self, _context: &str) {}
}
