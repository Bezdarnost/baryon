//! Container format detection from file signatures.

use std::fmt;

/// An image container format baryon knows how to recognize.
///
/// Recognizing a format does not imply it can be decoded; see [`ImageFormat::is_decodable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ImageFormat {
    /// JPEG (JFIF / EXIF).
    Jpeg,
    /// Portable Network Graphics.
    Png,
    /// JPEG XL, either a bare codestream or an ISOBMFF container.
    JpegXl,
    /// AV1 Image File Format.
    Avif,
}

impl ImageFormat {
    /// Detects the format from the leading bytes of a file.
    ///
    /// Only the signature is inspected, so this never fails on truncated data as long as the
    /// signature itself is present.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
        const JXL_CODESTREAM: &[u8] = &[0xFF, 0x0A];
        const JXL_CONTAINER: &[u8] = b"\x00\x00\x00\x0cJXL \r\n\x87\n";

        if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some(Self::Jpeg)
        } else if bytes.starts_with(PNG) {
            Some(Self::Png)
        } else if bytes.starts_with(JXL_CODESTREAM) || bytes.starts_with(JXL_CONTAINER) {
            Some(Self::JpegXl)
        } else if is_avif(bytes) {
            Some(Self::Avif)
        } else {
            None
        }
    }

    /// Canonical lowercase name, e.g. `"jpeg"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Png => "png",
            Self::JpegXl => "jxl",
            Self::Avif => "avif",
        }
    }

    /// Whether this build can decode the format.
    pub const fn is_decodable(self) -> bool {
        matches!(self, Self::Jpeg | Self::Png)
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Checks for an ISOBMFF `ftyp` box whose major or compatible brands include `avif`/`avis`.
fn is_avif(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let box_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let end = box_len.min(bytes.len());
    if end < 16 {
        return false;
    }
    // Layout: size(4) 'ftyp'(4) major_brand(4) minor_version(4) compatible_brands(4 * n).
    std::iter::once(&bytes[8..12])
        .chain(bytes[16..end].chunks_exact(4))
        .any(|brand| brand == b"avif" || brand == b"avis")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_signatures() {
        assert_eq!(
            ImageFormat::detect(&[0xFF, 0xD8, 0xFF, 0xE0]),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            ImageFormat::detect(b"\x89PNG\r\n\x1a\n\0\0"),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            ImageFormat::detect(&[0xFF, 0x0A, 0x00]),
            Some(ImageFormat::JpegXl)
        );
        assert_eq!(
            ImageFormat::detect(b"\x00\x00\x00\x0cJXL \r\n\x87\n\0"),
            Some(ImageFormat::JpegXl)
        );
    }

    #[test]
    fn detects_avif_by_major_and_compatible_brand() {
        let major = b"\x00\x00\x00\x14ftypavif\x00\x00\x00\x00mif1";
        assert_eq!(ImageFormat::detect(major), Some(ImageFormat::Avif));

        let compatible = b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00miafavif";
        assert_eq!(ImageFormat::detect(compatible), Some(ImageFormat::Avif));

        let heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic";
        assert_eq!(ImageFormat::detect(heic), None);
    }

    #[test]
    fn rejects_garbage_and_short_input() {
        assert_eq!(ImageFormat::detect(b""), None);
        assert_eq!(ImageFormat::detect(b"GIF89a"), None);
        assert_eq!(ImageFormat::detect(&[0xFF, 0xD8]), None);
    }
}
