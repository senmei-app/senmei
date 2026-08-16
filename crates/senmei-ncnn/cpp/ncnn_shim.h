#ifndef SENMEI_NCNN_SHIM_H
#define SENMEI_NCNN_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NcnnEngine NcnnEngine;

/* gpu: 1 = enable Vulkan, 0 = CPU. Returns NULL on failure. */
NcnnEngine* ncnn_engine_new(int gpu);

void ncnn_engine_destroy(NcnnEngine* engine);

/* Load a `.param` + `.bin` pair (same base name). Returns 0 on success. */
int ncnn_engine_load(NcnnEngine* engine, const char* param_path, const char* bin_path);

/* Run a single NCHW (1, 3, h, w) float input. On success allocates *output
 * (malloc, 3*oh*ow floats) and returns 0; the caller frees it with ncnn_free. */
int ncnn_engine_infer(NcnnEngine* engine, const float* input, int h, int w,
                      float** output, int* out_h, int* out_w);

void ncnn_free(float* p);

/* Last error message (valid until the next shim call). */
const char* ncnn_engine_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
