fn main() {
    #[cfg(feature = "hip")]
    hip::build();
}

#[cfg(feature = "hip")]
mod hip {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    pub fn build() {
        println!("cargo:rerun-if-changed=src/hip/kernels/");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");

        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
        let rocm_path = PathBuf::from(
            env::var("ROCM_PATH").expect("ROCM_PATH must be set for hip feature"),
        );

        // Compile HIP kernels
        compile_hip_kernels(&rocm_path, &out_dir);

        // Link settings
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=hip_kernels");
        println!(
            "cargo:rustc-link-search=native={}",
            rocm_path.join("lib").display()
        );
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
    }

    fn compile_hip_kernels(rocm_path: &PathBuf, out_dir: &PathBuf) {
        let hipcc = rocm_path.join("bin/hipcc");
        let kernel_dir = PathBuf::from("src/hip/kernels");

        let hip_files: Vec<PathBuf> = std::fs::read_dir(&kernel_dir)
            .expect("Failed to read kernel directory")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "hip"))
            .collect();

        if hip_files.is_empty() {
            panic!("No .hip files found in {}", kernel_dir.display());
        }

        let mut obj_files = Vec::new();

        for hip_file in &hip_files {
            println!("cargo:rerun-if-changed={}", hip_file.display());

            let stem = hip_file.file_stem().unwrap().to_str().unwrap();
            let obj_file = out_dir.join(format!("{}.o", stem));

            let status = Command::new(&hipcc)
                .args([
                    "-c",
                    "-fPIC",
                    "-O3",
                    "--offload-arch=gfx1151",
                    hip_file.to_str().unwrap(),
                    "-o",
                    obj_file.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to execute hipcc");

            if !status.success() {
                panic!("Failed to compile {}", hip_file.display());
            }

            obj_files.push(obj_file);
        }

        // Create static library
        let lib_path = out_dir.join("libhip_kernels.a");
        let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());

        let mut ar_cmd = Command::new(&ar);
        ar_cmd.arg("rcs").arg(&lib_path);
        for obj in &obj_files {
            ar_cmd.arg(obj);
        }

        let status = ar_cmd.status().expect("Failed to execute ar");
        if !status.success() {
            panic!("Failed to create static library");
        }
    }
}
