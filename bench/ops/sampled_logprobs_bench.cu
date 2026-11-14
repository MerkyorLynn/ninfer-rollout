#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/sampled_logprobs.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

int argument(int argc, char** argv, std::string_view name, int fallback) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            if (++i == argc) { throw std::invalid_argument(std::string(name) + " needs a value"); }
            return std::stoi(argv[i]);
        }
    }
    return fallback;
}

bool flag(int argc, char** argv, std::string_view name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) { return true; }
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no usable CUDA device");
        return 0;
    }
    try {
        const int batch   = argument(argc, argv, "--batch", 8);
        const int drafts  = argument(argc, argv, "--drafts", 7);
        const int columns = batch * (drafts + 1);
        const int extent  = argument(argc, argv, "--extent", drafts + 1);
        const int warmup  = argument(argc, argv, "--warmup", 12);
        const int repeat  = argument(argc, argv, "--repeat", 100);
        const int calls   = argument(argc, argv, "--graph-calls", 1);
        const bool reference = flag(argc, argv, "--reference");
        constexpr int rows  = 248320;
        constexpr int vocab = 248077;
        if (batch < 1 || batch > 8 || drafts < 1 || drafts > 15 || extent < 0 ||
            extent > drafts + 1 || warmup < 0 || repeat < 1 || calls < 1 || calls > 64) {
            throw std::invalid_argument("invalid benchmark dimensions or timing counts");
        }

        DeviceBuffer logits = make_bf16(static_cast<std::size_t>(rows) * columns, 23, -12.0F, 8.0F);
        DeviceBuffer tokens(sizeof(std::int32_t) * columns);
        DeviceBuffer maxima(sizeof(std::int32_t) * columns);
        DeviceBuffer counts(sizeof(std::int32_t) * batch);
        DeviceBuffer sampling(sizeof(float) * columns);
        DeviceBuffer raw(sizeof(float) * columns);
        Tensor t_logits(logits.p, DType::BF16, {rows, columns});
        Tensor t_tokens(tokens.p, DType::I32, {columns});
        Tensor t_maxima(maxima.p, DType::I32, {columns});
        Tensor t_counts(counts.p, DType::I32, {batch});
        Tensor t_sampling(sampling.p, DType::FP32, {columns});
        Tensor t_raw(raw.p, DType::FP32, {columns});
        std::vector<std::int32_t> host_tokens(columns);
        for (int i = 0; i < columns; ++i) { host_tokens[i] = (17 + i * 7919) % vocab; }
        CUDA_CHECK(cudaMemcpy(tokens.p, host_tokens.data(), tokens.bytes, cudaMemcpyHostToDevice));
        std::vector<std::int32_t> host_counts(batch, extent);
        CUDA_CHECK(cudaMemcpy(counts.p, host_counts.data(), counts.bytes, cudaMemcpyHostToDevice));
        std::vector<ops::SamplingConfig> host_configs(batch);
        for (auto& config : host_configs) {
            config.temperature = 0.6F;
            config.top_k       = 20;
            config.top_p       = 0.95F;
            config.seed        = 20261114;
        }
        DeviceBuffer configs(sizeof(ops::SamplingConfig) * batch);
        CUDA_CHECK(cudaMemcpy(configs.p, host_configs.data(), configs.bytes, cudaMemcpyHostToDevice));
        const auto* config_ptr = static_cast<const ops::SamplingConfig*>(configs.p);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        ops::argmax(t_logits, t_maxima, vocab, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto launch = [&](cudaStream_t target) {
            if (reference) {
                ops::sampled_token_logprobs(t_logits, t_tokens, &t_counts, config_ptr, nullptr,
                                            drafts + 1, vocab, t_sampling, t_raw, target);
            } else {
                ops::sampled_token_raw_logprobs(t_logits, t_tokens, t_maxima, &t_counts, drafts + 1,
                                                vocab, t_raw, target);
            }
        };
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        TimedGraph graph;
        graph.capture(stream, [&](cudaStream_t target) {
            for (int i = 0; i < calls; ++i) { launch(target); }
        });
        L2FlushBuffer flush(std::size_t{256} << 20);
        const ColdTiming timing = measure_cold_graph(graph, flush, stream, warmup, repeat);
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::printf("mode=%s B=%d K=%d licensed_per_row=%d columns=%d nodes=%zu calls=%d "
                    "median=%.3f us min=%.3f us p95=%.3f us\n",
                    reference ? "combined" : "raw_only", batch, drafts, extent, columns,
                    graph.nodes(), calls, timing.median_us / calls, timing.min_us / calls,
                    timing.p95_us / calls);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_sampled_logprobs_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
