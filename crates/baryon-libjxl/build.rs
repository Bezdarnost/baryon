//! Builds the vendored libjxl (encoder and its dependencies) as static libraries, plus the
//! small C shim in `shim/`, and links them into this crate.

use std::path::Path;

fn main() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("libjxl");
    assert!(
        source.join("third_party/highway/CMakeLists.txt").exists(),
        "libjxl sources are missing; run:\n  git submodule update --init --depth 1 \
         crates/baryon-libjxl/libjxl\n  git -C crates/baryon-libjxl/libjxl submodule update \
         --init --depth 1 third_party/highway third_party/brotli third_party/skcms"
    );

    let mut config = cmake::Config::new(&source);
    // Always optimize libjxl: a debug build encodes an order of magnitude slower.
    config.profile("Release");
    for (option, value) in [
        ("BUILD_SHARED_LIBS", "OFF"),
        ("BUILD_TESTING", "OFF"),
        ("CMAKE_POSITION_INDEPENDENT_CODE", "ON"),
        ("JPEGXL_ENABLE_BENCHMARK", "OFF"),
        ("JPEGXL_ENABLE_DEVTOOLS", "OFF"),
        ("JPEGXL_ENABLE_DOXYGEN", "OFF"),
        ("JPEGXL_ENABLE_EXAMPLES", "OFF"),
        ("JPEGXL_ENABLE_FUZZERS", "OFF"),
        ("JPEGXL_ENABLE_JNI", "OFF"),
        ("JPEGXL_ENABLE_MANPAGES", "OFF"),
        ("JPEGXL_ENABLE_OPENEXR", "OFF"),
        ("JPEGXL_ENABLE_PLUGINS", "OFF"),
        ("JPEGXL_ENABLE_SJPEG", "OFF"),
        ("JPEGXL_ENABLE_SKCMS", "ON"),
        ("JPEGXL_ENABLE_TCMALLOC", "OFF"),
        ("JPEGXL_ENABLE_TOOLS", "OFF"),
        ("JPEGXL_ENABLE_TRANSCODE_JPEG", "ON"),
        ("JPEGXL_ENABLE_VIEWERS", "OFF"),
        ("BROTLI_DISABLE_TESTS", "ON"),
        ("HWY_ENABLE_CONTRIB", "OFF"),
        ("HWY_ENABLE_EXAMPLES", "OFF"),
        ("HWY_ENABLE_INSTALL", "ON"),
        ("HWY_ENABLE_TESTS", "OFF"),
    ] {
        config.define(option, value);
    }
    if cfg!(target_env = "msvc") {
        // libjxl is written for Clang; match Rust's dynamic C runtime.
        config
            .generator_toolset("ClangCL")
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
    }
    let prefix = config.build();

    let lib_dir = ["lib", "lib64"]
        .iter()
        .map(|dir| prefix.join(dir))
        .find(|dir| dir.exists())
        .expect("libjxl build produced no lib directory");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    cc::Build::new()
        .file("shim/jxl_shim.c")
        .include(prefix.join("include"))
        .define("JXL_STATIC_DEFINE", None)
        .define("JXL_CMS_STATIC_DEFINE", None)
        .warnings(true)
        .compile("baryon_jxl_shim");

    for lib in [
        "jxl",
        "jxl_cms",
        "hwy",
        "brotlienc",
        "brotlidec",
        "brotlicommon",
    ] {
        println!("cargo:rustc-link-lib=static={}", static_name(&lib_dir, lib));
    }
    link_cpp_runtime();

    println!("cargo:rerun-if-changed=shim/jxl_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Static libraries are named with a `-static` suffix in some libjxl/brotli configurations.
fn static_name(lib_dir: &Path, name: &str) -> String {
    let candidates = [format!("{name}-static"), name.to_owned()];
    candidates
        .into_iter()
        .find(|candidate| {
            [format!("lib{candidate}.a"), format!("{candidate}.lib")]
                .iter()
                .any(|file| lib_dir.join(file).exists())
        })
        .unwrap_or_else(|| panic!("static library {name} not found in {}", lib_dir.display()))
}

fn link_cpp_runtime() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if !target.contains("msvc") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
