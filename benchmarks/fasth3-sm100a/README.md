# FastH3 8-Step V2 on GB200

This branch contains the vLLM-Omni implementation and frozen qualification artifacts for `FastVideo/FastVideo-FastH3-8-Step-V2@3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, based on upstream `bcb4ae288f77dbee431c0649b72ef43a4722bb2b`. The [upstream model recipe](../../recipes/MiniMaxAI/MiniMax-H3.md#fasth3-8-step-v2-full-checkpoint) loads the full V2 checkpoint with the `fl2va` serving partition and no additional LoRA.

The H3 attention implementation honors the released 80% video sparsity, preserves prefix attention and trained compression gates, and rejects a missing sparse layout. The shared sparse dispatcher records the actual kernel provider and avoids retaining GPU tensors through cached exception tracebacks. `FASTVIDEO_VSA_SM100A=1` requests the native FastVideo kernel; check the provider log because the upstream dispatcher can fall back if the native extension fails.

## Frozen evidence

The [canonical run](artifacts/blog-fast-h3-8step-20260925/baseline/vllm-omni/20260926T041709Z/) retains the exact TOML and resolved plan, source snapshot, environment, kernel build provenance, server logs, raw request records, media inspection and all twelve original MP4s. [FILES.sha256](FILES.sha256) covers the exact evidence bytes. The implementation is committed directly in this repository; archived patches preserve the identity of the measured working tree.

Qualification uses four GB200 GPUs, BF16, Ulysses 4, text encoder TP1, tiled VAE patch parallelism 4, resident components and regional compilation. Stage profiling is enabled. The fixed workload is 1344×768 at 24 FPS, concurrency one, six excluded warmups and six disjoint measured requests: 5/10/15 seconds × 1K/10K prompt tokens. All twelve outputs passed full H.264/AAC decoding, frame counts, stereo 32 kHz audio, audiovisual alignment and nondegenerate media checks. Measured complete-response latencies in that order were 7.90, 11.83, 16.41, 22.13, 27.65 and 35.96 seconds. Eight attention regression tests passed, including independent GPU reference comparisons for both providers; the original validation log is included.

These are qualification observations, not a formal latency/throughput campaign or a backend performance ranking. Native encoding differs across backends. Eight-GPU deployments are not qualified, and SM120 cannot execute this SM100a kernel.

## Reproduce

1. Install this branch with its `vsa` extra and match the recorded environment, including vLLM 0.30.0, Python 3.12, PyTorch 2.13.0+cu130 and CUDA 13.0 on ARM64 GB200. Provision the pinned checkpoint and its referenced base components.
2. Build the exact optional kernel source against that environment:

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
PATH=/usr/local/cuda-13.0/bin:$PATH \
TORCH_CUDA_ARCH_LIST=10.0a CMAKE_BUILD_PARALLEL_LEVEL=8 \
uv pip install --python /path/to/backend-env/bin/python \
  --no-deps --no-build-isolation --no-binary fastvideo-kernel \
  --reinstall-package fastvideo-kernel fastvideo-kernel==0.3.4
```

The source distribution SHA-256 is `e6c3fb2d4e9615d07ac81de3bc821e3c9224bc21e9d83be734e0cefcb379ef5c`. The measured library SHA-256 is `bbdb129f92f1cff8c91728c4bb3205d458064730e9080e5610efe7e5b7566733`. The included reproduction wheel is that tested ARM64/Python 3.12 build; do not use it on a different architecture or ABI.

3. Prepare the evaluator from `Boreas618/uniserve@9458bca425a96d5bc45f5e97da9d2986449fad3e`, apply `artifacts/blog-fast-h3-8step-20260925/reproduction/uniserve.patch`, and install its locked GPU environment. The reproduction directory also retains the evaluator source, lockfile and source identity. Copy this directory's `artifacts/` tree into that evaluator checkout without replacing existing runs.
4. The frozen profile is `artifacts/blog-fast-h3-8step-20260925/baseline/vllm-omni/20260926T041709Z/sm100a.toml`. Preserve the recorded `/workspace/UniServe` layout, putting this branch at `refs/vllm-omni`, or copy the profile and consistently relocate the repository, environment, checkpoint, dataset and output paths. Keep the request manifests, numerical settings and load protocol unchanged.
5. From the evaluator checkout, resolve `.venv/bin/uniserve-eval --config <profile> plan vllm-omni-bf16-qualification-c1`, then execute the matching `run vllm-omni-bf16-qualification-c1 --output-root <fresh-directory>`. Run only one measurement point at a time and retain warmups and original response bodies. Require actual SM100a dispatch with no native-provider failure for an eligible SM100a result.

Prepared campaign profiles and corpus authoring inputs are included for reproducibility; their presence does not certify unexecuted campaign points.
