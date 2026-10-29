/* A small, flat C interface over libjxl's encoder for baryon's Rust bindings.
 *
 * Keeping all libjxl structs on the C side means the Rust code only declares the functions
 * below, so struct layouts can never drift between the two languages.
 *
 * Every function returns 0 on success, or a libjxl JxlEncoderError code (> 0), or a negative
 * shim error, and fills *out / *out_len with a buffer the caller frees with
 * baryon_jxl_free(). Work is distributed through the given libjxl parallel runner.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/parallel_runner.h>

#define SHIM_ERR_ALLOC (-1)
#define SHIM_ERR_ARGUMENT (-2)
#define SHIM_ERR_NO_JPEG (-3)
#define SHIM_ERR_DECODE (-4)

static int error_code(JxlEncoder* enc) {
  int code = (int)JxlEncoderGetError(enc);
  return code == 0 ? (int)JXL_ENC_ERR_GENERIC : code;
}

/* Collects the encoded stream into a growing malloc'd buffer. */
static int drain(JxlEncoder* enc, uint8_t** out, size_t* out_len) {
  size_t capacity = 1 << 16;
  uint8_t* buffer = (uint8_t*)malloc(capacity);
  if (buffer == NULL) return SHIM_ERR_ALLOC;
  uint8_t* next = buffer;
  size_t available = capacity;
  for (;;) {
    JxlEncoderStatus status = JxlEncoderProcessOutput(enc, &next, &available);
    if (status == JXL_ENC_SUCCESS) break;
    if (status != JXL_ENC_NEED_MORE_OUTPUT) {
      free(buffer);
      return error_code(enc);
    }
    size_t used = (size_t)(next - buffer);
    capacity *= 2;
    uint8_t* grown = (uint8_t*)realloc(buffer, capacity);
    if (grown == NULL) {
      free(buffer);
      return SHIM_ERR_ALLOC;
    }
    buffer = grown;
    next = buffer + used;
    available = capacity - used;
  }
  *out = buffer;
  *out_len = (size_t)(next - buffer);
  return 0;
}

static JxlEncoder* create(JxlParallelRunner runner, void* runner_opaque) {
  JxlEncoder* enc = JxlEncoderCreate(NULL);
  if (enc != NULL && runner != NULL &&
      JxlEncoderSetParallelRunner(enc, runner, runner_opaque) != JXL_ENC_SUCCESS) {
    JxlEncoderDestroy(enc);
    return NULL;
  }
  return enc;
}

int baryon_jxl_encode_jpeg(const uint8_t* jpeg, size_t jpeg_len, int effort,
                           JxlParallelRunner runner, void* runner_opaque,
                           uint8_t** out, size_t* out_len) {
  JxlEncoder* enc = create(runner, runner_opaque);
  if (enc == NULL) return SHIM_ERR_ALLOC;
  int result = 0;
  JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(enc, NULL);
  if (settings == NULL ||
      JxlEncoderStoreJPEGMetadata(enc, JXL_TRUE) != JXL_ENC_SUCCESS ||
      JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort) !=
          JXL_ENC_SUCCESS ||
      JxlEncoderAddJPEGFrame(settings, jpeg, jpeg_len) != JXL_ENC_SUCCESS) {
    result = error_code(enc);
  } else {
    JxlEncoderCloseInput(enc);
    result = drain(enc, out, out_len);
  }
  JxlEncoderDestroy(enc);
  return result;
}

int baryon_jxl_encode_pixels(const void* pixels, size_t pixels_len, uint32_t width,
                             uint32_t height, uint32_t channels, uint32_t bits_per_sample,
                             int lossless, float distance, int effort,
                             JxlParallelRunner runner, void* runner_opaque, uint8_t** out,
                             size_t* out_len) {
  if (channels < 1 || channels > 4 || (bits_per_sample != 8 && bits_per_sample != 16)) {
    return SHIM_ERR_ARGUMENT;
  }
  int gray = channels <= 2;
  int alpha = channels == 2 || channels == 4;

  JxlEncoder* enc = create(runner, runner_opaque);
  if (enc == NULL) return SHIM_ERR_ALLOC;

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = width;
  info.ysize = height;
  info.bits_per_sample = bits_per_sample;
  info.num_color_channels = gray ? 1 : 3;
  info.num_extra_channels = alpha ? 1 : 0;
  info.alpha_bits = alpha ? bits_per_sample : 0;
  /* Lossless coding must keep the samples as they are, without the XYB colour transform. */
  info.uses_original_profile = lossless ? JXL_TRUE : JXL_FALSE;

  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, gray ? JXL_TRUE : JXL_FALSE);

  JxlPixelFormat format = {channels, bits_per_sample == 8 ? JXL_TYPE_UINT8 : JXL_TYPE_UINT16,
                           JXL_NATIVE_ENDIAN, 0};

  int result = 0;
  JxlEncoderFrameSettings* settings = NULL;
  if (JxlEncoderSetBasicInfo(enc, &info) != JXL_ENC_SUCCESS ||
      JxlEncoderSetColorEncoding(enc, &color) != JXL_ENC_SUCCESS) {
    result = error_code(enc);
  }
  if (result == 0 && alpha) {
    JxlExtraChannelInfo extra;
    JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA, &extra);
    extra.bits_per_sample = bits_per_sample;
    if (JxlEncoderSetExtraChannelInfo(enc, 0, &extra) != JXL_ENC_SUCCESS) {
      result = error_code(enc);
    }
  }
  if (result == 0) {
    settings = JxlEncoderFrameSettingsCreate(enc, NULL);
    if (settings == NULL ||
        JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort) !=
            JXL_ENC_SUCCESS ||
        (lossless ? JxlEncoderSetFrameLossless(settings, JXL_TRUE)
                  : JxlEncoderSetFrameDistance(settings, distance)) != JXL_ENC_SUCCESS ||
        JxlEncoderAddImageFrame(settings, &format, pixels, pixels_len) != JXL_ENC_SUCCESS) {
      result = error_code(enc);
    }
  }
  if (result == 0) {
    JxlEncoderCloseInput(enc);
    result = drain(enc, out, out_len);
  }
  JxlEncoderDestroy(enc);
  return result;
}

/* Restores the JPEG stored in a JPEG XL file by lossless transcoding. Returns
 * SHIM_ERR_NO_JPEG if the file holds no JPEG reconstruction data, SHIM_ERR_DECODE if libjxl
 * cannot decode it. */
int baryon_jxl_reconstruct_jpeg(const uint8_t* jxl, size_t jxl_len, JxlParallelRunner runner,
                                void* runner_opaque, uint8_t** out, size_t* out_len) {
  JxlDecoder* dec = JxlDecoderCreate(NULL);
  if (dec == NULL) return SHIM_ERR_ALLOC;
  int result = 0;
  size_t capacity = jxl_len + jxl_len / 2 + 4096;
  uint8_t* buffer = (uint8_t*)malloc(capacity);
  int have_jpeg = 0;
  if (buffer == NULL) {
    result = SHIM_ERR_ALLOC;
  } else if ((runner != NULL &&
              JxlDecoderSetParallelRunner(dec, runner, runner_opaque) != JXL_DEC_SUCCESS) ||
             JxlDecoderSubscribeEvents(dec, JXL_DEC_JPEG_RECONSTRUCTION | JXL_DEC_FULL_IMAGE) !=
                 JXL_DEC_SUCCESS ||
             JxlDecoderSetInput(dec, jxl, jxl_len) != JXL_DEC_SUCCESS) {
    result = SHIM_ERR_DECODE;
  } else {
    JxlDecoderCloseInput(dec);
    for (;;) {
      JxlDecoderStatus status = JxlDecoderProcessInput(dec);
      if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
        have_jpeg = 1;
        if (JxlDecoderSetJPEGBuffer(dec, buffer, capacity) != JXL_DEC_SUCCESS) {
          result = SHIM_ERR_DECODE;
          break;
        }
      } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
        size_t used = capacity - JxlDecoderReleaseJPEGBuffer(dec);
        capacity *= 2;
        uint8_t* grown = (uint8_t*)realloc(buffer, capacity);
        if (grown == NULL) {
          result = SHIM_ERR_ALLOC;
          break;
        }
        buffer = grown;
        if (JxlDecoderSetJPEGBuffer(dec, buffer + used, capacity - used) != JXL_DEC_SUCCESS) {
          result = SHIM_ERR_DECODE;
          break;
        }
      } else if (status == JXL_DEC_FULL_IMAGE) {
        /* The JPEG is complete once its frame has been processed. */
        if (have_jpeg) {
          *out_len = capacity - JxlDecoderReleaseJPEGBuffer(dec);
          *out = buffer;
          buffer = NULL;
          break;
        }
      } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
        /* Pixels are about to be decoded, so there is no JPEG to reconstruct. */
        result = SHIM_ERR_NO_JPEG;
        break;
      } else if (status == JXL_DEC_SUCCESS) {
        result = SHIM_ERR_NO_JPEG;
        break;
      } else {
        result = SHIM_ERR_DECODE;
        break;
      }
    }
  }
  free(buffer);
  JxlDecoderDestroy(dec);
  return result;
}

void baryon_jxl_free(uint8_t* buffer) { free(buffer); }
