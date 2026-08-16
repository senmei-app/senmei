#include "ncnn_shim.h"

#include <ncnn/net.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}
}  // namespace

struct NcnnEngine {
    ncnn::Net net;
    ncnn::Option opt;
    bool gpu = false;
    std::string in_name;
    std::string out_name;
};

NcnnEngine* ncnn_engine_new(int gpu) {
    ncnn::create_gpu_instance();
    auto* engine = new NcnnEngine();
    engine->gpu = gpu != 0 && ncnn::get_gpu_count() > 0;
    engine->opt = ncnn::Option();
    engine->opt.use_vulkan_compute = engine->gpu;
    engine->opt.use_fp16_packed = engine->gpu;
    engine->opt.use_fp16_storage = engine->gpu;
    engine->opt.use_fp16_arithmetic = engine->gpu;
    engine->opt.num_threads = 4;
    engine->net.opt = engine->opt;
    return engine;
}

void ncnn_engine_destroy(NcnnEngine* engine) {
    delete engine;
    ncnn::destroy_gpu_instance();
}

int ncnn_engine_load(NcnnEngine* engine, const char* param, const char* bin) {
    engine->net.opt = engine->opt;
    int rp = engine->net.load_param(param);
    if (rp != 0) {
        set_error("load_param failed: " + std::to_string(rp));
        return -1;
    }
    int rb = engine->net.load_model(bin);
    if (rb != 0) {
        set_error("load_model failed: " + std::to_string(rb));
        return -1;
    }
    // Detect input/output blob names so the shim works for any single-input,
    // single-output model (the blob names differ per model). `Layer::tops` hold
    // blob indices, not names.
    const auto& blobs = engine->net.blobs();
    for (const auto* layer : engine->net.layers()) {
        if (layer->type == std::string("Input") && !layer->tops.empty()) {
            int idx = layer->tops[0];
            if (idx >= 0 && idx < (int)blobs.size()) {
                engine->in_name = blobs[idx].name;
            }
            break;
        }
    }
    const auto& layers = engine->net.layers();
    if (!layers.empty() && !layers.back()->tops.empty()) {
        int idx = layers.back()->tops[0];
        if (idx >= 0 && idx < (int)blobs.size()) {
            engine->out_name = blobs[idx].name;
        }
    }
    if (engine->in_name.empty() || engine->out_name.empty()) {
        set_error("could not detect input/output blobs");
        return -1;
    }
    return 0;
}

int ncnn_engine_infer(NcnnEngine* engine, const float* input, int h, int w,
                      float** output, int* out_h, int* out_w) {
    ncnn::Mat in(w, h, 3, const_cast<float*>(input));
    ncnn::Extractor ex = engine->net.create_extractor();
    ncnn::Mat out;
    if (ex.input(engine->in_name.c_str(), in) != 0) {
        set_error("extractor input failed for blob '" + engine->in_name + "'");
        return -1;
    }
    if (ex.extract(engine->out_name.c_str(), out) != 0) {
        set_error("extractor output failed for blob '" + engine->out_name + "'");
        return -1;
    }
    *out_h = out.h;
    *out_w = out.w;
    size_t total = (size_t)out.c * out.h * out.w;
    auto* buf = (float*)std::malloc(total * sizeof(float));
    if (!buf) {
        set_error("out of memory");
        return -1;
    }
    for (int c = 0; c < out.c; c++) {
        const float* src = (const float*)out.channel(c);
        std::memcpy(buf + (size_t)c * out.h * out.w, src, sizeof(float) * out.h * out.w);
    }
    *output = buf;
    return 0;
}

void ncnn_free(float* p) {
    std::free(p);
}

const char* ncnn_engine_last_error() {
    return g_last_error.c_str();
}
