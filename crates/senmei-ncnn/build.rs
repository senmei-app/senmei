use std::env;
use std::path::PathBuf;
use std::process::Command;

const NCNN_TAG: &str = "20260526";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/ncnn_shim.h");
    println!("cargo:rerun-if-changed=cpp/ncnn_shim.cpp");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ncnn = manifest.join("../../third_party/ncnn");
    let build = ncnn.join("build");

    if !ncnn.join("src/net.h").exists() {
        run(
            "git",
            &[
                "clone",
                "--depth",
                "1",
                "--branch",
                NCNN_TAG,
                "https://github.com/Tencent/ncnn",
                ncnn.to_str().unwrap(),
            ],
        );
        run(
            "git",
            &[
                "-C",
                ncnn.to_str().unwrap(),
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
            ],
        );
    }
    if !build.join("src/libncnn.a").exists() {
        run(
            "cmake",
            &[
                "-S",
                ncnn.to_str().unwrap(),
                "-B",
                build.to_str().unwrap(),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DNCNN_VULKAN=ON",
                "-DNCNN_BUILD_TOOLS=OFF",
                "-DNCNN_BUILD_EXAMPLES=OFF",
                "-DNCNN_BUILD_BENCHMARK=OFF",
                "-DNCNN_BUILD_TESTS=OFF",
                "-DNCNN_BUILD_SHARED_LIB=OFF",
                "-DNCNN_BUILD_STATIC_LIB=ON",
            ],
        );
        run(
            "cmake",
            &[
                "--build",
                build.to_str().unwrap(),
                "-j",
                &parallelism(),
            ],
        );
    }

    // Compile the shim first so its static lib precedes -lncnn in the link
    // line (static libs only resolve symbols from libs listed to their right).
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(ncnn.join("src"))
        .file("cpp/ncnn_shim.cpp")
        .flag_if_supported("-fopenmp")
        .warnings(false)
        .compile("senmei_ncnn_shim");

    println!("cargo:rustc-link-search=native={}", build.join("src").display());
    for sub in [
        "glslang/glslang",
        "glslang/SPIRV",
        "glslang/glslang/OSDependent/Unix",
    ] {
        println!("cargo:rustc-link-search=native={}", build.join(sub).display());
    }
    println!("cargo:rustc-link-lib=static=ncnn");
    println!("cargo:rustc-link-lib=static=glslang");
    println!("cargo:rustc-link-lib=static=MachineIndependent");
    println!("cargo:rustc-link-lib=static=GenericCodeGen");
    println!("cargo:rustc-link-lib=static=SPIRV");
    println!("cargo:rustc-link-lib=static=OSDependent");
    println!("cargo:rustc-link-lib=static=glslang-default-resource-limits");
    println!("cargo:rustc-link-lib=vulkan");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=pthread");

    let bindings = bindgen::Builder::default()
        .header("cpp/ncnn_shim.h")
        .allowlist_function("ncnn_.*")
        .allowlist_type("NcnnEngine")
        .generate()
        .expect("bindgen failed");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}

fn parallelism() -> String {
    std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".into())
}

fn run(program: &str, args: &[&str]) {
    let status = Command::new(program)
        .args(args)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {program}: {e}"));
    assert!(status.success(), "{program} failed: {args:?}");
}
