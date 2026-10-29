//! The one EXIF field baryon needs: the orientation tag.
//!
//! EXIF data is a TIFF structure: a byte-order mark, the number 42, and the offset of the first
//! image file directory (IFD0), whose 12-byte entries hold the tags. Orientation (tag 0x0112)
//! lives in IFD0. Every read is bounds-checked; malformed data yields `None`.

use crate::geometry::Orientation;

const ORIENTATION_TAG: u16 = 0x0112;
const SHORT: u16 = 3;

/// Reads the orientation from EXIF data starting at the TIFF header.
///
/// Returns `None` when the data is malformed, the tag is absent or its value is out of range.
pub(crate) fn orientation(tiff: &[u8]) -> Option<Orientation> {
    let big_endian = match tiff.get(..2)? {
        b"II" => false,
        b"MM" => true,
        _ => return None,
    };
    let u16_at = |offset: usize| -> Option<u16> {
        let bytes: [u8; 2] = tiff.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(if big_endian {
            u16::from_be_bytes(bytes)
        } else {
            u16::from_le_bytes(bytes)
        })
    };
    let u32_at = |offset: usize| -> Option<u32> {
        let bytes: [u8; 4] = tiff.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
        Some(if big_endian {
            u32::from_be_bytes(bytes)
        } else {
            u32::from_le_bytes(bytes)
        })
    };

    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = usize::try_from(u32_at(4)?).ok()?;
    let entries = u16_at(ifd)?;
    (0..usize::from(entries)).find_map(|index| {
        let entry = ifd + 2 + index * 12;
        if u16_at(entry)? != ORIENTATION_TAG || u16_at(entry + 2)? != SHORT {
            return None;
        }
        Orientation::from_exif(u16_at(entry + 8)?)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::geometry::Rotation;

    /// A minimal EXIF block holding only the orientation tag.
    pub(crate) fn exif_with_orientation(value: u16, big_endian: bool) -> Vec<u8> {
        let u16b = |v: u16| {
            if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let u32b = |v: u32| {
            if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let mut out = Vec::new();
        out.extend(if big_endian { b"MM" } else { b"II" });
        out.extend(u16b(42));
        out.extend(u32b(8));
        // IFD0 with one unrelated tag before the orientation, then no next IFD.
        out.extend(u16b(2));
        out.extend(u16b(0x010F)); // Make, ASCII, count 4, inline "baz\0"
        out.extend(u16b(2));
        out.extend(u32b(4));
        out.extend(b"baz\0");
        out.extend(u16b(ORIENTATION_TAG));
        out.extend(u16b(SHORT));
        out.extend(u32b(1));
        out.extend(u16b(value));
        out.extend(u16b(0));
        out.extend(u32b(0));
        out
    }

    #[test]
    fn reads_both_byte_orders() {
        for big_endian in [false, true] {
            let exif = exif_with_orientation(6, big_endian);
            assert_eq!(
                orientation(&exif),
                Some(Orientation::IDENTITY.then_rotate(Rotation::Cw90))
            );
        }
    }

    #[test]
    fn rejects_malformed_data() {
        let exif = exif_with_orientation(3, false);
        for len in 0..exif.len() - 6 {
            assert_eq!(orientation(&exif[..len]), None, "truncated to {len} bytes");
        }
        assert_eq!(orientation(&exif_with_orientation(0, false)), None);
        assert_eq!(orientation(&exif_with_orientation(9, true)), None);
        let mut bad_magic = exif.clone();
        bad_magic[2] = 43;
        assert_eq!(orientation(&bad_magic), None);
        let mut huge_offset = exif;
        huge_offset[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(orientation(&huge_offset), None);
    }
}
