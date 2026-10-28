//! Container format detection from file signatures.

use std::fmt;

/// The kind of media a container holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MediaType {
    /// Still images.
    Image,
}

/// A container format baryon recognizes from its signature.
///
/// Recognizing a format does not imply it can be decoded; the media crates report which
/// formats they support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// JPEG (JFIF / EXIF).
    Jpeg,
    /// Portable Network Graphics.
    Png,
    /// JPEG XL, either a bare codestream or an ISOBMFF container.
    JpegXl,
    /// AV1 Image File Format.
    Avif,
}

impl Format {
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

    /// The kind of media this format holds.
    pub const fn media_type(self) -> MediaType {
        match self {
            Self::Jpeg | Self::Png | Self::JpegXl | Self::Avif => MediaType::Image,
        }
    }
}

impl fmt::Display for Format {
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
        .chain(
            bytes[16..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(<[u8; 4]>::as_slice),
        )
        .any(|brand| brand == b"avif" || brand == b"avis")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_signatures() {
        assert_eq!(
            Format::detect(&[0xFF, 0xD8, 0xFF, 0xE0]),
            Some(Format::Jpeg)
        );
        assert_eq!(Format::detect(b"\x89PNG\r\n\x1a\n\0\0"), Some(Format::Png));
        assert_eq!(Format::detect(&[0xFF, 0x0A, 0x00]), Some(Format::JpegXl));
        assert_eq!(
            Format::detect(b"\x00\x00\x00\x0cJXL \r\n\x87\n\0"),
            Some(Format::JpegXl)
        );
    }

    #[test]
    fn detects_avif_by_major_and_compatible_brand() {
        let major = b"\x00\x00\x00\x14ftypavif\x00\x00\x00\x00mif1";
        assert_eq!(Format::detect(major), Some(Format::Avif));

        let compatible = b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00miafavif";
        assert_eq!(Format::detect(compatible), Some(Format::Avif));

        let heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic";
        assert_eq!(Format::detect(heic), None);
    }

    #[test]
    fn rejects_garbage_and_short_input() {
        assert_eq!(Format::detect(b""), None);
        assert_eq!(Format::detect(b"GIF89a"), None);
        assert_eq!(Format::detect(&[0xFF, 0xD8]), None);
    }

    #[test]
    fn every_format_is_an_image_today() {
        for format in [Format::Jpeg, Format::Png, Format::JpegXl, Format::Avif] {
            assert_eq!(format.media_type(), MediaType::Image);
        }
    }
}
