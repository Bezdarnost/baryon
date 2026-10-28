//! Python bindings for `baryon-core`, exposed as the `baryon._baryon` extension module.
//!
//! The Python-facing `Image` is an immutable handle to a lazy pipeline: every transformation
//! returns a new handle, and pixels are only produced by `to_numpy()` / `numpy.asarray()`. All
//! decoding and processing runs with the GIL released, on a process-wide thread pool.

// Doc comments here become Python docstrings, written in reST/Google style rather than rustdoc.
#![allow(clippy::doc_markdown)]

use std::path::PathBuf;
use std::sync::{PoisonError, RwLock};

use baryon_core::{Error, Executor, Image, LazyImage, PixelBuffer, Rect, ResizeFilter, Rotation};
use numpy::IntoPyArray;
use numpy::ndarray::{Array3, ShapeError};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, intern};
use rayon::prelude::*;

create_exception!(
    baryon,
    BaryonError,
    PyException,
    "Base class for every error raised by baryon."
);
create_exception!(
    baryon,
    DecodeError,
    BaryonError,
    "The image data is corrupt or could not be decoded."
);
create_exception!(
    baryon,
    UnsupportedFormatError,
    BaryonError,
    "The image format is not recognized or not supported by this build."
);

fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::UnrecognizedFormat | Error::Unsupported { .. } => {
            UnsupportedFormatError::new_err(err.to_string())
        }
        Error::Decode { .. } => DecodeError::new_err(err.to_string()),
        Error::InvalidArgument(message) => PyValueError::new_err(message),
        Error::Io(io) => io.into(),
        _ => BaryonError::new_err(err.to_string()),
    }
}

/// Converts the error for item `index` of a batch, keeping its exception type and prefixing
/// the message with the position, e.g. ``images[3]: failed to decode ...``.
fn to_py_err_at(py: Python<'_>, what: &str, index: usize, err: Error) -> PyErr {
    let err = to_py_err(err);
    let message = format!("{what}[{index}]: {}", err.value(py));
    match err.get_type(py).call1((message,)) {
        Ok(value) => PyErr::from_value(value),
        Err(failed) => failed,
    }
}

/// The process-wide executor, tagged with the id of the process that created it.
static EXECUTOR: RwLock<Option<(u32, Executor)>> = RwLock::new(None);

/// Returns the shared executor, creating it on first use.
///
/// Thread pools do not survive `fork()`: a child process inherits the pool but not its threads,
/// so using it would hang. This matters for PyTorch `DataLoader` workers and `multiprocessing`
/// on Linux, so a forked child transparently gets a fresh pool.
fn executor() -> PyResult<Executor> {
    let pid = std::process::id();
    if let Some((owner, executor)) = EXECUTOR
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        && *owner == pid
    {
        return Ok(executor.clone());
    }
    let mut slot = EXECUTOR.write().unwrap_or_else(PoisonError::into_inner);
    match slot.take() {
        Some((owner, executor)) if owner == pid => {
            *slot = Some((owner, executor.clone()));
            return Ok(executor);
        }
        // Inherited across fork: its worker threads do not exist here, and dropping it could
        // block on them, so it is leaked deliberately.
        Some((_, stale)) => std::mem::forget(stale),
        None => {}
    }
    let executor = Executor::new(threads_from_env()?).map_err(to_py_err)?;
    *slot = Some((pid, executor.clone()));
    Ok(executor)
}

fn threads_from_env() -> PyResult<usize> {
    match std::env::var("BARYON_NUM_THREADS") {
        Err(_) => Ok(0),
        Ok(value) => value.trim().parse().map_err(|_| {
            PyValueError::new_err(format!(
                "BARYON_NUM_THREADS must be a non-negative integer, got {value:?}"
            ))
        }),
    }
}

/// Sets the number of threads baryon uses for decoding and processing.
///
/// The setting covers everything baryon runs: batch functions such as :func:`to_numpy_all`
/// spread images across threads, and a single large image is split across threads where the
/// format and operations allow it (JPEG XL decoding, resize, crop, rotate and flip).
///
/// The initial value comes from the ``BARYON_NUM_THREADS`` environment variable, defaulting to
/// one thread per logical CPU.
///
/// Args:
///     num_threads: Number of threads; ``None`` or ``0`` means one per logical CPU.
#[pyfunction]
#[pyo3(signature = (num_threads = None))]
fn set_num_threads(num_threads: Option<usize>) -> PyResult<()> {
    let executor = Executor::new(num_threads.unwrap_or(0)).map_err(to_py_err)?;
    *EXECUTOR.write().unwrap_or_else(PoisonError::into_inner) =
        Some((std::process::id(), executor));
    Ok(())
}

/// Returns the number of threads baryon uses; see :func:`set_num_threads`.
#[pyfunction]
fn get_num_threads() -> PyResult<usize> {
    Ok(executor()?.num_threads())
}

/// Where to read an image from.
enum Input {
    /// Encoded bytes from any object supporting the buffer protocol.
    Data(Vec<u8>),
    /// A filesystem path.
    Path(PathBuf),
}

impl Input {
    fn open(self) -> baryon_core::Result<LazyImage> {
        match self {
            Self::Data(data) => LazyImage::from_bytes(data),
            Self::Path(path) => LazyImage::open(path),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Input {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // Buffers are tried first: `os.fspath` accepts `bytes`, which would otherwise be
        // misread as a path.
        if let Ok(buffer) = PyBuffer::<u8>::get(&obj) {
            return Ok(Self::Data(buffer.to_vec(obj.py())?));
        }
        obj.extract::<PathBuf>().map(Self::Path).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a path (str or os.PathLike) or a bytes-like object, got {}",
                obj.get_type()
                    .name()
                    .map_or_else(|_| "?".into(), |n| n.to_string())
            ))
        })
    }
}

/// Opens an image lazily.
///
/// Only the header is parsed; no pixels are decoded until the result is converted to an array.
///
/// Args:
///     source: A file path (``str`` or ``os.PathLike``) or the encoded image as a bytes-like
///         object.
///
/// Raises:
///     UnsupportedFormatError: The format is not recognized or not supported.
///     DecodeError: The header is corrupt.
///     OSError: The file could not be read.
#[pyfunction]
fn open(py: Python<'_>, source: Input) -> PyResult<PyImage> {
    let inner = py.detach(|| source.open()).map_err(to_py_err)?;
    Ok(PyImage { inner })
}

/// Opens many images lazily, reading files and parsing headers in parallel.
///
/// Equivalent to ``[open(s) for s in sources]``, but faster for many files.
///
/// Args:
///     sources: File paths and/or bytes-like objects, as accepted by :func:`open`.
///
/// Raises:
///     The first error any source raises in :func:`open`, with its position prefixed to the
///     message, e.g. ``sources[3]: ...``.
#[pyfunction]
fn open_all(py: Python<'_>, sources: Vec<Input>) -> PyResult<Vec<PyImage>> {
    let executor = executor()?;
    let opened: Vec<_> =
        py.detach(|| executor.install(|| sources.into_par_iter().map(Input::open).collect()));
    opened
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result
                .map(PyImage::from)
                .map_err(|err| to_py_err_at(py, "sources", index, err))
        })
        .collect()
}

/// Decodes many images in parallel and returns their pixels.
///
/// Equivalent to ``[image.to_numpy() for image in images]``, but runs on all of baryon's
/// threads (see :func:`set_num_threads`) with the GIL released once for the whole batch.
///
/// Args:
///     images: Images from :func:`open` or :func:`open_all`, with any pending transformations.
///
/// Returns:
///     One ``(height, width, channels)`` array per image, in input order.
///
/// Raises:
///     The first error any image raises in :meth:`Image.to_numpy`, with its position prefixed
///     to the message, e.g. ``images[3]: ...``.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts sequence arguments into an owned Vec"
)]
fn to_numpy_all<'py>(
    py: Python<'py>,
    images: Vec<Bound<'py, PyImage>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let executor = executor()?;
    let pipelines: Vec<LazyImage> = images
        .iter()
        .map(|image| image.get().inner.clone())
        .collect();
    let decoded = py.detach(|| executor.execute_all(&pipelines));
    decoded
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            let image = result.map_err(|err| to_py_err_at(py, "images", index, err))?;
            into_ndarray(py, image)
        })
        .collect()
}

/// A lazily decoded image.
///
/// Transformations such as :meth:`crop` and :meth:`resize` return a new ``Image`` and do no
/// work; they are optimized and executed together when pixels are requested with
/// :meth:`to_numpy` or ``numpy.asarray(image)``.
#[pyclass(module = "baryon", name = "Image", frozen)]
struct PyImage {
    inner: LazyImage,
}

#[pymethods]
impl PyImage {
    /// Output width in pixels, after all pending transformations.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Output height in pixels, after all pending transformations.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// ``(width, height)`` after all pending transformations.
    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.inner.width(), self.inner.height())
    }

    /// Shape of the array :meth:`to_numpy` returns: ``(height, width, channels)``.
    #[getter]
    fn shape(&self) -> (u32, u32, usize) {
        (self.inner.height(), self.inner.width(), self.channels())
    }

    /// Number of channels: 1 (``L``), 2 (``LA``), 3 (``RGB``) or 4 (``RGBA``).
    #[getter]
    fn channels(&self) -> usize {
        self.inner.color().channels()
    }

    /// Channel layout: ``"L"``, ``"LA"``, ``"RGB"`` or ``"RGBA"``.
    #[getter]
    fn mode(&self) -> &'static str {
        self.inner.color().name()
    }

    /// NumPy dtype name of the pixel data: ``"uint8"`` or ``"uint16"``.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.sample_type().name()
    }

    /// Source container format, e.g. ``"jpeg"`` or ``"png"``.
    #[getter]
    fn format(&self) -> &'static str {
        self.inner.info().format.name()
    }

    /// ``(width, height)`` of the encoded source image, before any transformation.
    #[getter]
    fn source_size(&self) -> (u32, u32) {
        let info = self.inner.info();
        (info.width, info.height)
    }

    /// Whether the source is stored progressively (progressive JPEG, interlaced PNG), which
    /// limits how much work partial decoding can skip.
    #[getter]
    fn progressive(&self) -> bool {
        self.inner.info().progressive
    }

    /// Returns a new image limited to the given rectangle.
    ///
    /// Coordinates are relative to this image, after its pending transformations. When the crop
    /// comes before any resize it is pushed down into the decoder.
    ///
    /// Raises:
    ///     ValueError: The rectangle is empty or extends past the image.
    fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> PyResult<Self> {
        self.inner
            .crop(Rect::new(x, y, width, height))
            .map(Self::from)
            .map_err(to_py_err)
    }

    /// Returns a new image resized to exactly ``width`` x ``height``.
    ///
    /// Args:
    ///     width: Target width in pixels.
    ///     height: Target height in pixels.
    ///     filter: One of ``"nearest"``, ``"box"``, ``"bilinear"``, ``"hamming"``,
    ///         ``"bicubic"``, ``"mitchell"`` or ``"lanczos3"``.
    ///
    /// Raises:
    ///     ValueError: A dimension is zero or the filter is unknown.
    #[pyo3(signature = (width, height, *, filter = "lanczos3"))]
    fn resize(&self, width: u32, height: u32, filter: &str) -> PyResult<Self> {
        let filter: ResizeFilter = filter.parse().map_err(to_py_err)?;
        self.inner
            .resize(width, height, filter)
            .map(Self::from)
            .map_err(to_py_err)
    }

    /// Returns a new image rotated by a multiple of 90 degrees, **clockwise** by default.
    ///
    /// Note:
    ///     Pillow's ``Image.rotate`` and ``numpy.rot90`` turn counter-clockwise for positive
    ///     angles. baryon turns clockwise, like CSS and photo viewers. When porting code from
    ///     those libraries, pass ``clockwise=False`` to keep their behaviour.
    ///
    /// Args:
    ///     degrees: Angle in degrees; any multiple of 90, including negative values.
    ///     clockwise: Rotate clockwise (default) or counter-clockwise.
    ///
    /// Raises:
    ///     ValueError: ``degrees`` is not a multiple of 90.
    #[pyo3(signature = (degrees, *, clockwise = true))]
    fn rotate(&self, degrees: i64, clockwise: bool) -> PyResult<Self> {
        let rotation = Rotation::from_degrees(degrees).map_err(PyValueError::new_err)?;
        let rotation = if clockwise {
            rotation
        } else {
            rotation.map(Rotation::inverse)
        };
        Ok(match rotation {
            Some(rotation) => self.inner.rotate(rotation).into(),
            None => self.inner.clone().into(),
        })
    }

    /// Returns a new image mirrored left to right.
    fn flip_horizontal(&self) -> Self {
        self.inner.flip_horizontal().into()
    }

    /// Returns a new image mirrored top to bottom.
    fn flip_vertical(&self) -> Self {
        self.inner.flip_vertical().into()
    }

    /// Describes the pipeline.
    ///
    /// Args:
    ///     optimized: Show the physical plan that will run (default) instead of the
    ///         transformations as recorded.
    #[pyo3(signature = (*, optimized = true))]
    fn explain(&self, optimized: bool) -> String {
        self.inner.explain(optimized)
    }

    /// Decodes the image, applies all pending transformations and returns the pixels.
    ///
    /// The array always has shape ``(height, width, channels)`` and dtype ``uint8`` or
    /// ``uint16``; see :attr:`shape` and :attr:`dtype`. The GIL is released while decoding, and
    /// large images use several threads where possible (see :func:`set_num_threads`). For many
    /// images, :func:`to_numpy_all` is faster.
    ///
    /// Raises:
    ///     DecodeError: The image data is corrupt.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let executor = executor()?;
        let image = py
            .detach(|| executor.execute(&self.inner))
            .map_err(to_py_err)?;
        into_ndarray(py, image)
    }

    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Every call materializes a fresh array, so `copy` only matters for the dtype cast.
        let array = self.to_numpy(py)?;
        match dtype {
            Some(dtype) if !array.getattr(intern!(py, "dtype"))?.eq(&dtype)? => {
                if copy == Some(false) {
                    return Err(PyValueError::new_err(
                        "converting to the requested dtype requires a copy",
                    ));
                }
                array.call_method1(intern!(py, "astype"), (dtype,))
            }
            _ => Ok(array),
        }
    }

    fn __repr__(&self) -> String {
        let pending = self.inner.ops().len();
        format!(
            "<baryon.Image {} {}x{} {} {}{}>",
            self.format(),
            self.inner.width(),
            self.inner.height(),
            self.mode(),
            self.dtype(),
            match pending {
                0 => String::new(),
                1 => ", 1 pending op".to_owned(),
                n => format!(", {n} pending ops"),
            }
        )
    }
}

impl From<LazyImage> for PyImage {
    fn from(inner: LazyImage) -> Self {
        Self { inner }
    }
}

/// Moves decoded pixels into a NumPy array without copying.
fn into_ndarray(py: Python<'_>, image: Image) -> PyResult<Bound<'_, PyAny>> {
    let shape = (
        image.height() as usize,
        image.width() as usize,
        image.color().channels(),
    );
    let shape_err = |err: ShapeError| BaryonError::new_err(format!("internal error: {err}"));
    Ok(match image.into_pixels() {
        PixelBuffer::U8(data) => Array3::from_shape_vec(shape, data)
            .map_err(shape_err)?
            .into_pyarray(py)
            .into_any(),
        PixelBuffer::U16(data) => Array3::from_shape_vec(shape, data)
            .map_err(shape_err)?
            .into_pyarray(py)
            .into_any(),
    })
}

/// The native extension module behind the `baryon` package.
#[pymodule(gil_used = false)]
fn _baryon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyImage>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(open_all, m)?)?;
    m.add_function(wrap_pyfunction!(to_numpy_all, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add("BaryonError", py.get_type::<BaryonError>())?;
    m.add("DecodeError", py.get_type::<DecodeError>())?;
    m.add(
        "UnsupportedFormatError",
        py.get_type::<UnsupportedFormatError>(),
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
