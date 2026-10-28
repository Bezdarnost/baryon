//! Rectangles and the dihedral orientation group used by the optimizer.

use std::fmt;

/// An axis-aligned rectangle in pixel coordinates. Always non-empty when produced by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Rect {
    /// Creates a rectangle.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// A rectangle anchored at the origin.
    pub const fn from_size(width: u32, height: u32) -> Self {
        Self::new(0, 0, width, height)
    }

    /// Right edge (exclusive). Computed in `u64` so it cannot overflow.
    pub const fn right(&self) -> u64 {
        self.x as u64 + self.width as u64
    }

    /// Bottom edge (exclusive). Computed in `u64` so it cannot overflow.
    pub const fn bottom(&self) -> u64 {
        self.y as u64 + self.height as u64
    }

    /// Whether `inner` lies entirely inside `self`.
    pub const fn contains(&self, inner: &Rect) -> bool {
        inner.x >= self.x
            && inner.y >= self.y
            && inner.right() <= self.right()
            && inner.bottom() <= self.bottom()
    }

    /// Expresses `inner`, given in the same coordinate space as `self`, relative to `self`'s origin.
    ///
    /// The caller must ensure `self.contains(inner)`.
    pub(crate) fn relative(&self, inner: &Rect) -> Rect {
        debug_assert!(self.contains(inner));
        Rect::new(
            inner.x - self.x,
            inner.y - self.y,
            inner.width,
            inner.height,
        )
    }
}

impl fmt::Display for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}+{}+{}", self.width, self.height, self.x, self.y)
    }
}

/// One of the eight lossless orientations of a rectangular grid (the dihedral group D4).
///
/// It is stored as a mapping from *output* coordinates to *source* coordinates: an output pixel
/// `(x, y)` is first transposed to `(y, x)` if `transpose` is set, and then each axis is mirrored
/// if the matching flip flag is set. Any sequence of flips and quarter rotations collapses into
/// one of these, which lets the optimizer move crops in front of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Orientation {
    transpose: bool,
    flip_x: bool,
    flip_y: bool,
}

impl Orientation {
    /// The orientation that leaves the image unchanged.
    pub const IDENTITY: Self = Self {
        transpose: false,
        flip_x: false,
        flip_y: false,
    };

    /// Whether this orientation leaves the image unchanged.
    pub const fn is_identity(self) -> bool {
        !self.transpose && !self.flip_x && !self.flip_y
    }

    /// Whether output width and height are swapped relative to the source.
    pub const fn swaps_axes(self) -> bool {
        self.transpose
    }

    /// Output dimensions for a source of the given size.
    pub const fn output_size(self, width: u32, height: u32) -> (u32, u32) {
        if self.transpose {
            (height, width)
        } else {
            (width, height)
        }
    }

    /// Composes a horizontal mirror of the current output onto this orientation.
    #[must_use]
    pub const fn then_flip_horizontal(mut self) -> Self {
        if self.transpose {
            self.flip_y = !self.flip_y;
        } else {
            self.flip_x = !self.flip_x;
        }
        self
    }

    /// Composes a vertical mirror of the current output onto this orientation.
    #[must_use]
    pub const fn then_flip_vertical(mut self) -> Self {
        if self.transpose {
            self.flip_x = !self.flip_x;
        } else {
            self.flip_y = !self.flip_y;
        }
        self
    }

    /// Composes a clockwise quarter turn of the current output onto this orientation.
    #[must_use]
    pub const fn then_rotate(self, rotation: Rotation) -> Self {
        match rotation {
            Rotation::Cw90 => self.then_rotate_cw90(),
            Rotation::Cw180 => self.then_flip_horizontal().then_flip_vertical(),
            Rotation::Cw270 => self
                .then_rotate_cw90()
                .then_flip_horizontal()
                .then_flip_vertical(),
        }
    }

    const fn then_rotate_cw90(mut self) -> Self {
        // New output (x, y) reads old output (y, H - 1 - x).
        if self.transpose {
            self.flip_x = !self.flip_x;
        } else {
            self.flip_y = !self.flip_y;
        }
        self.transpose = !self.transpose;
        self
    }

    /// Maps a rectangle in output coordinates back to source coordinates.
    ///
    /// `source_width`/`source_height` are the dimensions of the oriented source region.
    pub fn source_rect(self, output: Rect, source_width: u32, source_height: u32) -> Rect {
        let (mut x, mut y, w, h) = if self.transpose {
            (output.y, output.x, output.height, output.width)
        } else {
            (output.x, output.y, output.width, output.height)
        };
        if self.flip_x {
            x = source_width - x - w;
        }
        if self.flip_y {
            y = source_height - y - h;
        }
        Rect::new(x, y, w, h)
    }

    /// Maps a single output pixel to the source pixel it is read from.
    pub(crate) const fn source_pixel(
        self,
        x: u32,
        y: u32,
        source_width: u32,
        source_height: u32,
    ) -> (u32, u32) {
        let (mut u, mut v) = if self.transpose { (y, x) } else { (x, y) };
        if self.flip_x {
            u = source_width - 1 - u;
        }
        if self.flip_y {
            v = source_height - 1 - v;
        }
        (u, v)
    }
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return f.write_str("identity");
        }
        let parts: Vec<&str> = [
            (self.transpose, "transpose"),
            (self.flip_x, "flip_x"),
            (self.flip_y, "flip_y"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect();
        f.write_str(&parts.join("+"))
    }
}

/// A clockwise rotation by a multiple of 90 degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// 90 degrees clockwise.
    Cw90,
    /// 180 degrees.
    Cw180,
    /// 270 degrees clockwise (90 counter-clockwise).
    Cw270,
}

impl Rotation {
    /// Parses a clockwise angle in degrees. Accepts any multiple of 90, including negatives.
    ///
    /// Returns `Ok(None)` for angles equivalent to no rotation.
    pub fn from_degrees(degrees: i64) -> Result<Option<Self>, String> {
        if degrees % 90 != 0 {
            return Err(format!(
                "rotation must be a multiple of 90 degrees, got {degrees}"
            ));
        }
        Ok(match degrees.rem_euclid(360) {
            90 => Some(Self::Cw90),
            180 => Some(Self::Cw180),
            270 => Some(Self::Cw270),
            _ => None,
        })
    }

    /// The rotation that undoes this one.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Cw90 => Self::Cw270,
            Self::Cw180 => Self::Cw180,
            Self::Cw270 => Self::Cw90,
        }
    }

    /// Clockwise angle in degrees.
    pub const fn degrees(self) -> u32 {
        match self {
            Self::Cw90 => 90,
            Self::Cw180 => 180,
            Self::Cw270 => 270,
        }
    }
}

impl fmt::Display for Rotation {
    /// Spells out the direction, since conventions differ between libraries.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cw90 => "90° clockwise",
            Self::Cw180 => "180°",
            Self::Cw270 => "90° counter-clockwise",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_ROTATIONS: [Rotation; 3] = [Rotation::Cw90, Rotation::Cw180, Rotation::Cw270];

    #[test]
    fn group_identities_hold() {
        let id = Orientation::IDENTITY;
        assert!(
            id.then_flip_horizontal()
                .then_flip_horizontal()
                .is_identity()
        );
        assert!(id.then_flip_vertical().then_flip_vertical().is_identity());
        let four_turns = (0..4).fold(id, |o, _| o.then_rotate(Rotation::Cw90));
        assert!(four_turns.is_identity());
        assert!(
            id.then_rotate(Rotation::Cw90)
                .then_rotate(Rotation::Cw270)
                .is_identity()
        );
        assert_eq!(
            id.then_rotate(Rotation::Cw180),
            id.then_rotate(Rotation::Cw90).then_rotate(Rotation::Cw90)
        );
    }

    #[test]
    fn rotate_cw90_maps_top_left_to_top_right() {
        // A 3x2 source rotated clockwise is 2x3; output top-right (1, 0) shows source (0, 0).
        let o = Orientation::IDENTITY.then_rotate(Rotation::Cw90);
        assert_eq!(o.output_size(3, 2), (2, 3));
        assert_eq!(o.source_pixel(1, 0, 3, 2), (0, 0));
        assert_eq!(o.source_pixel(0, 0, 3, 2), (0, 1));
    }

    #[test]
    fn source_rect_agrees_with_source_pixel() {
        let (sw, sh) = (7, 5);
        let mut orientations = vec![Orientation::IDENTITY];
        for r in ALL_ROTATIONS {
            orientations.push(Orientation::IDENTITY.then_rotate(r));
            orientations.push(Orientation::IDENTITY.then_rotate(r).then_flip_horizontal());
        }
        orientations.push(Orientation::IDENTITY.then_flip_horizontal());

        for o in orientations {
            let (ow, oh) = o.output_size(sw, sh);
            let out = Rect::new(1, 2, ow - 2, oh - 3);
            let src = o.source_rect(out, sw, sh);
            for y in out.y..out.y + out.height {
                for x in out.x..out.x + out.width {
                    let (u, v) = o.source_pixel(x, y, sw, sh);
                    assert!(
                        src.contains(&Rect::new(u, v, 1, 1)),
                        "{o}: ({x},{y}) -> ({u},{v}) outside {src}"
                    );
                }
            }
            assert_eq!(
                u64::from(src.width) * u64::from(src.height),
                u64::from(out.width) * u64::from(out.height)
            );
        }
    }

    #[test]
    fn inverse_undoes_rotation() {
        for r in ALL_ROTATIONS {
            let o = Orientation::IDENTITY
                .then_rotate(r)
                .then_rotate(r.inverse());
            assert!(o.is_identity(), "{r} then its inverse");
        }
    }

    #[test]
    fn parses_rotation_degrees() {
        assert_eq!(Rotation::from_degrees(90), Ok(Some(Rotation::Cw90)));
        assert_eq!(Rotation::from_degrees(-90), Ok(Some(Rotation::Cw270)));
        assert_eq!(Rotation::from_degrees(720), Ok(None));
        assert!(Rotation::from_degrees(45).is_err());
    }
}
