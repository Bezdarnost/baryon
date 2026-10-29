//! JPEG XL encoding with the vendored reference encoder, libjxl.
//!
//! This is the only crate in baryon with `unsafe` code. libjxl is reached through a small C shim
//! (`shim/jxl_shim.c`) with three flat functions, so no C struct layouts are mirrored in Rust.
//! Everything here is wrapped in a safe API.
//!
//! libjxl's parallel work runs on the current rayon pool: called from inside a baryon
//! `Executor`, encoding uses that executor's threads; called from anywhere else, it runs on the
//! calling thread.

use std::ffi::{c_int, c_void};
use std::fmt;

use rayon::prelude::*;

/// libjxl's `JxlParallelRunInit`.
type RunInit = unsafe extern "C" fn(opaque: *mut c_void, num_threads: usize) -> c_int;
/// libjxl's `JxlParallelRunFunction`.
type RunFunction = unsafe extern "C" fn(opaque: *mut c_void, value: u32, thread_id: usize);
/// libjxl's `JxlParallelRunner`.
type Runner = unsafe extern "C" fn(
    runner_opaque: *mut c_void,
    jpegxl_opaque: *mut c_void,
    init: RunInit,
    func: RunFunction,
    start: u32,
    end: u32,
) -> c_int;

unsafe extern "C" {
    fn baryon_jxl_encode_jpeg(
        jpeg: *const u8,
        jpeg_len: usize,
        effort: c_int,
        runner: Option<Runner>,
        runner_opaque: *mut c_void,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;

    fn baryon_jxl_encode_pixels(
        pixels: *const c_void,
        pixels_len: usize,
        width: u32,
        height: u32,
        channels: u32,
        bits_per_sample: u32,
        lossless: c_int,
        distance: f32,
        effort: c_int,
        runner: Option<Runner>,
        runner_opaque: *mut c_void,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;

    fn baryon_jxl_reconstruct_jpeg(
        jxl: *const u8,
        jxl_len: usize,
        runner: Option<Runner>,
        runner_opaque: *mut c_void,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;

    fn baryon_jxl_free(buffer: *mut u8);
}

/// An encoding failure reported by libjxl or the shim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: c_int,
}

impl Error {
    /// Whether the JPEG XL file holds no JPEG to reconstruct, because it was not made by
    /// lossless JPEG transcoding.
    pub const fn is_missing_jpeg(self) -> bool {
        self.code == -3
    }

    /// Whether libjxl rejected a JPEG it cannot transcode losslessly (an unusual JPEG
    /// structure), as opposed to a general failure. Such files can still be encoded from their
    /// decoded pixels.
    pub const fn is_untranscodable_jpeg(self) -> bool {
        // JXL_ENC_ERR_GENERIC and JXL_ENC_ERR_JBRD are what AddJPEGFrame reports for JPEGs
        // whose structure it cannot represent.
        matches!(self.code, 1 | 3)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self.code {
            1 => "libjxl reported a generic error",
            2 => "libjxl ran out of memory",
            3 => "the JPEG cannot be transcoded losslessly",
            4 => "libjxl rejected the input as invalid",
            0x80 => "the requested encoding is not supported by libjxl",
            0x81 => "libjxl reported an API usage error (this is a bug in baryon)",
            -1 => "memory allocation failed",
            -2 => "unsupported pixel layout",
            -3 => "the file holds no JPEG reconstruction data",
            -4 => "libjxl could not decode the file",
            _ => "unknown libjxl error",
        };
        write!(f, "{message} (code {})", self.code)
    }
}

impl std::error::Error for Error {}

/// How pixels are compressed by [`encode_pixels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quality {
    /// Mathematically lossless.
    Lossless,
    /// Lossy, at the given Butteraugli distance: 0.0 to 25.0, where 1.0 is visually lossless.
    Distance(f32),
}

/// Encoder effort, 1 (fastest) to 10 (smallest files). libjxl's default is 7.
pub const DEFAULT_EFFORT: u8 = 7;

/// Re-encodes a JPEG file as JPEG XL without any loss. The original JPEG can be reconstructed
/// bit for bit from the result.
pub fn encode_jpeg(jpeg: &[u8], effort: u8) -> Result<Vec<u8>, Error> {
    let effort = checked_effort(effort)?;
    let mut out = std::ptr::null_mut();
    let mut out_len = 0;
    // SAFETY: `jpeg` is valid for `jpeg.len()` bytes for the whole call, the runner is a valid
    // `JxlParallelRunner` that ignores its opaque pointer, and `out`/`out_len` are valid
    // places for the shim to write its result to.
    let code = unsafe {
        baryon_jxl_encode_jpeg(
            jpeg.as_ptr(),
            jpeg.len(),
            effort,
            Some(rayon_runner),
            std::ptr::null_mut(),
            &raw mut out,
            &raw mut out_len,
        )
    };
    take_output(code, out, out_len)
}

/// Encodes interleaved pixels: 1 (gray), 2 (gray + alpha), 3 (RGB) or 4 (RGBA) channels of
/// 8-bit or native-endian 16-bit samples, in sRGB.
pub fn encode_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    bits_per_sample: u32,
    quality: Quality,
    effort: u8,
) -> Result<Vec<u8>, Error> {
    let effort = checked_effort(effort)?;
    let expected =
        u64::from(width) * u64::from(height) * u64::from(channels) * u64::from(bits_per_sample / 8);
    if !(1..=4).contains(&channels)
        || !matches!(bits_per_sample, 8 | 16)
        || pixels.len() as u64 != expected
    {
        return Err(Error { code: -2 });
    }
    let (lossless, distance) = match quality {
        Quality::Lossless => (1, 0.0),
        Quality::Distance(distance) => (0, distance),
    };
    let mut out = std::ptr::null_mut();
    let mut out_len = 0;
    // SAFETY: `pixels` holds exactly `width * height * channels` samples of the declared size
    // (checked above) and stays valid for the whole call; the runner and output pointers are
    // valid as in `encode_jpeg`.
    let code = unsafe {
        baryon_jxl_encode_pixels(
            pixels.as_ptr().cast(),
            pixels.len(),
            width,
            height,
            channels,
            bits_per_sample,
            lossless,
            distance,
            effort,
            Some(rayon_runner),
            std::ptr::null_mut(),
            &raw mut out,
            &raw mut out_len,
        )
    };
    take_output(code, out, out_len)
}

/// Restores the original JPEG file from a JPEG XL file made by [`encode_jpeg`] (or any other
/// lossless JPEG transcoder), bit for bit.
pub fn reconstruct_jpeg(jxl: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = std::ptr::null_mut();
    let mut out_len = 0;
    // SAFETY: `jxl` is valid for `jxl.len()` bytes for the whole call; the runner and output
    // pointers are valid as in `encode_jpeg`.
    let code = unsafe {
        baryon_jxl_reconstruct_jpeg(
            jxl.as_ptr(),
            jxl.len(),
            Some(rayon_runner),
            std::ptr::null_mut(),
            &raw mut out,
            &raw mut out_len,
        )
    };
    take_output(code, out, out_len)
}

fn checked_effort(effort: u8) -> Result<c_int, Error> {
    if (1..=10).contains(&effort) {
        Ok(c_int::from(effort))
    } else {
        Err(Error { code: -2 })
    }
}

/// Copies the shim's output into a `Vec` and frees the C buffer.
fn take_output(code: c_int, out: *mut u8, out_len: usize) -> Result<Vec<u8>, Error> {
    if code != 0 {
        return Err(Error { code });
    }
    // SAFETY: on success the shim returns a `malloc`ed buffer of `out_len` initialized bytes,
    // which is read once here and then released with the shim's matching `free`.
    let bytes = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
    // SAFETY: `out` came from the shim's allocator and is not used after this call.
    unsafe { baryon_jxl_free(out) };
    Ok(bytes)
}

/// A raw pointer that libjxl guarantees may be shared across its worker callbacks.
#[derive(Clone, Copy)]
struct SharedOpaque(*mut c_void);

// SAFETY: libjxl's parallel runner contract allows `func` to be called concurrently from
// several threads with the same `jpegxl_opaque` pointer, as long as thread ids are distinct.
unsafe impl Send for SharedOpaque {}
// SAFETY: as above.
unsafe impl Sync for SharedOpaque {}

impl SharedOpaque {
    const fn get(self) -> *mut c_void {
        self.0
    }
}

/// A `JxlParallelRunner` that executes libjxl's work items on the current rayon pool.
///
/// Thread ids are rayon worker indices, which are distinct and below the pool size that is
/// reported to `init`. Off a rayon worker, the work runs sequentially on the calling thread.
unsafe extern "C" fn rayon_runner(
    _runner_opaque: *mut c_void,
    jpegxl_opaque: *mut c_void,
    init: RunInit,
    func: RunFunction,
    start: u32,
    end: u32,
) -> c_int {
    let parallel = rayon::current_thread_index().is_some() && rayon::current_num_threads() > 1;
    let threads = if parallel {
        rayon::current_num_threads()
    } else {
        1
    };
    // SAFETY: libjxl passes a valid `init` and the opaque pointer it expects back.
    let ret = unsafe { init(jpegxl_opaque, threads) };
    if ret != 0 {
        return ret;
    }
    let opaque = SharedOpaque(jpegxl_opaque);
    if parallel {
        (start..end).into_par_iter().for_each(|value| {
            let thread = rayon::current_thread_index().unwrap_or(0);
            // SAFETY: `func` is valid for this runner call; `thread` is the index of the
            // rayon worker running it, unique among concurrent calls and below `threads`.
            unsafe { func(opaque.get(), value, thread) };
        });
    } else {
        for value in start..end {
            // SAFETY: as above, with all work on one thread using thread id 0.
            unsafe { func(opaque.get(), value, 0) };
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: u32, height: u32, channels: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    (0..channels)
                        .map(move |c| u8::try_from((x * 7 + y * 3 + c * 50) % 256).unwrap())
                })
            })
            .collect()
    }

    #[test]
    fn encodes_every_layout() {
        for channels in 1..=4 {
            let pixels = gradient(64, 48, channels);
            let jxl = encode_pixels(&pixels, 64, 48, channels, 8, Quality::Lossless, 3).unwrap();
            assert_eq!(&jxl[..2], &[0xFF, 0x0A], "bare codestream signature");
        }
    }

    #[test]
    fn sixteen_bit_and_lossy_encode() {
        let samples: Vec<u16> = (0..32u32 * 32 * 3)
            .map(|v| u16::try_from(v * 61 % 65_536).unwrap())
            .collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();
        assert!(encode_pixels(&bytes, 32, 32, 3, 16, Quality::Lossless, 1).is_ok());
        let pixels = gradient(64, 64, 3);
        assert!(encode_pixels(&pixels, 64, 64, 3, 8, Quality::Distance(1.0), 3).is_ok());
    }

    fn jpeg(width: u16, height: u16) -> Vec<u8> {
        let pixels = gradient(u32::from(width), u32::from(height), 3);
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 90)
            .encode(&pixels, width, height, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        out
    }

    /// The same JPEG with a comment (COM) segment right after SOI, a structure some JPEG XL
    /// decoders reconstruct incorrectly.
    fn with_comment(jpeg: &[u8]) -> Vec<u8> {
        let comment = b"CREATOR: baryon test";
        let length = u16::try_from(comment.len() + 2).unwrap().to_be_bytes();
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xFE, length[0], length[1]]);
        out.extend_from_slice(comment);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn jpeg_transcoding_round_trips_bit_exactly() {
        let plain = jpeg(96, 64);
        for jpeg in [plain.clone(), with_comment(&plain)] {
            let jxl = encode_jpeg(&jpeg, 7).unwrap();
            assert_eq!(reconstruct_jpeg(&jxl).unwrap(), jpeg);
        }
    }

    #[test]
    fn pixel_encoded_files_have_no_jpeg() {
        let pixels = gradient(16, 16, 3);
        let jxl = encode_pixels(&pixels, 16, 16, 3, 8, Quality::Lossless, 1).unwrap();
        let err = reconstruct_jpeg(&jxl).unwrap_err();
        assert!(err.is_missing_jpeg(), "{err}");
        assert!(reconstruct_jpeg(b"\xff\x0a garbage").is_err());
    }

    #[test]
    fn rejects_bad_arguments() {
        let pixels = gradient(8, 8, 3);
        assert!(encode_pixels(&pixels, 8, 8, 4, 8, Quality::Lossless, 3).is_err());
        assert!(encode_pixels(&pixels, 8, 8, 3, 12, Quality::Lossless, 3).is_err());
        assert!(encode_pixels(&pixels, 8, 8, 3, 8, Quality::Lossless, 0).is_err());
        assert!(encode_jpeg(b"not a jpeg", 7).is_err());
    }

    #[test]
    fn parallel_and_sequential_output_are_identical() {
        let pixels = gradient(700, 500, 3);
        let sequential = encode_pixels(&pixels, 700, 500, 3, 8, Quality::Lossless, 3).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| encode_pixels(&pixels, 700, 500, 3, 8, Quality::Lossless, 3))
            .unwrap();
        assert_eq!(sequential, parallel);
    }
}
