"""Freeze launch candidates and serial evaluator profiles for the campaign."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/workspace/UniServe')
CAMPAIGN = ROOT / 'artifacts/blog-fast-h3-8step-20260925'
PROTOCOL = CAMPAIGN / 'protocol/workload/manifests'
PROFILES = CAMPAIGN / 'profiles/workload'
PROFILES.mkdir(parents=True, exist_ok=False)
MODEL = '/workspace/models/FastVideo-FastH3-8-Step-V2'


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def main():
    lock = {'sources': {}, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'request_timeout_s': 21600, 'poll_interval_s': 0.1, 'connection_reuse': 'unbounded persistent HTTP/1.1 pool', 'checkpoints': {}}
    for name, path in [('uniserve', ROOT), ('fastvideo', ROOT / 'refs/FastVideo'), ('vllm-omni', ROOT / 'refs/vllm-omni'), ('sglang', ROOT / 'refs/sglang-current')]:
        patch = subprocess.check_output(['git', '-C', str(path), 'diff', 'HEAD', '--binary'])
        (CAMPAIGN / f'protocol/workload/{name}.patch').write_bytes(patch)
        lock['sources'][name] = {'path': str(path), 'revision': git(path, 'rev-parse', 'HEAD'), 'status': git(path, 'status', '--short'), 'patch_sha256': hashlib.sha256(patch).hexdigest()}
    lock['evaluator_files'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'uniserve_eval').rglob('*.py')}
    snapshot = CAMPAIGN / 'protocol/workload/evaluator-source'
    snapshot.mkdir(exist_ok=False)
    for name in lock['evaluator_files']:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    lock['evaluator_snapshot'] = str(snapshot)
    for tier, directory in [('bf16', Path(MODEL)), ('nvfp4', Path(MODEL + '-NVFP4'))]:
        revision = (directory / '.cache/huggingface/download/README.md.metadata').read_text().splitlines()[0]
        lock['checkpoints'][tier] = {'path': str(directory), 'revision': revision, 'contract': json.loads((directory / 'fastvideo_inference.json').read_text()), 'content_index_sha256': hashlib.sha256((directory / 'checkpoint_content.json').read_bytes()).hexdigest()}
    (CAMPAIGN / 'protocol/workload/source-lock.json').write_text(json.dumps(lock, indent=2) + '\n')

    for hardware, gpus, deployment in [('gb200-4', 4, 'configs/minimax-h3-four-devices.json'), ('gb200-8', 8, 'configs/minimax-h3-eight-devices.json'), ('rtxpro6000-8', 8, 'configs/minimax-h3-eight-devices-single-node.json')]:
        servers = {}
        for tier in ('bf16', 'nvfp4'):
            model = MODEL + ('-NVFP4' if tier == 'nvfp4' else '')
            servers[f'uniserve-{tier}'] = ['target/release/uniserve', 'serve', model, '--served-model-name', 'FastH3', '--host', '127.0.0.1', '--port', '18083', '--worker-python', '.venv/bin/python', '--workers', deployment, '--max-batch', '2', '--max-running-requests', '2', '--max-num-batched-tokens', '2', '--chunked-prefill-size', '1', '--max-model-len', '16384', '--max-video-seconds', '15', '--mem-fraction-static', '0.70', '--dtype', 'bfloat16']
        # Direct native services: no proxy performs encoding or muxing for them.
        yaml_path = PROFILES / f'fastvideo-{hardware}.yaml'
        config = f'''generator:
  model_path: {MODEL}
  engine:
    num_gpus: {gpus}
    execution_backend: {"ray" if hardware == "gb200-8" else "mp"}
    use_fsdp_inference: false
    parallelism:
      tp_size: 1
      sp_size: {gpus}
    offload:
      dit: false
      text_encoder: false
      vae: false
    compile:
      enabled: false
      vae_enabled: true
  pipeline:
    workload_type: t2v
    experimental:
      attention_backend: VIDEO_SPARSE_ATTN_H3
      VSA_sparsity: 0.8
      VSA_tile_size: 64
      inference_torch_compile: true
      vae_parallel_decode: true
      vae_parallel_decode_strategy: gather
server:
  host: 127.0.0.1
  port: 18083
  output_dir: {CAMPAIGN}/qualification/server-media/{hardware}/fastvideo
  served_model_name: FastH3
default_request:
  negative_prompt: ""
  sampling:
    height: 768
    width: 1344
    num_frames: 124
    fps: 24
    num_inference_steps: 9
    guidance_scale: 1.0
    batch_cfg: false
  output:
    return_frames: false
'''
        yaml_path.write_text(config)
        servers['fastvideo-bf16'] = ['.venv-fastvideo/bin/fastvideo', 'serve', '--config', str(yaml_path)]
        servers['vllm-omni-bf16'] = ['.venv-vllm-omni/bin/vllm', 'serve', MODEL, '--served-model-name', 'FastH3', '--omni', '--task-type', 'fl2va', '--trust-remote-code', '--host', '127.0.0.1', '--port', '18083', '--num-gpus', str(gpus), '--usp', str(gpus), '--ring', '1', '--vae-patch-parallel-size', str(gpus), '--vae-parallel-mode', 'tile', '--vae-use-tiling', '--diffusion-attention-backend', 'FASTVIDEO_VSA']
        servers['sglang-bf16'] = ['/workspace/sglang-0.5.17-env/bin/sglang', 'serve', '--model-type', 'diffusion', '--model-path', MODEL, '--num-gpus', str(gpus), '--ulysses-degree', str(gpus), '--performance-mode', 'speed', '--encoder-parallel', 'auto', '--attention-backend', 'fa', '--component-attention-backends', 'transformer=VIDEO_SPARSE_ATTN_H3', '--attention-backend-config', '{"VSA_sparsity":0.8,"vsa_tile_size":64}', '--output-path', str(CAMPAIGN / 'qualification/server-media' / hardware / 'sglang'), '--host', '127.0.0.1', '--port', '18083']
        servers['sglang-bf16'] += ['--served-model-name', 'FastH3']
        if hardware == 'gb200-8':
            for tier in ('bf16', 'nvfp4'):
                command = servers[f'uniserve-{tier}']
                command += ['--host-identity', 'rank-0']
                servers[f'uniserve-{tier}'] = ['.venv/bin/python', str(CAMPAIGN / 'protocol/serve_two_hosts.py'), *command]
            servers['vllm-omni-bf16'] += ['--worker-backend', 'ray', '--ray-address', 'auto']
            # Multi-host SGLang deployment has not been qualified. A validated
            # single-host command does not establish two-host launch semantics.
            del servers['sglang-bf16']
        lines = [f'root = {json.dumps(str(ROOT))}', f'artifact_root = {json.dumps(str(CAMPAIGN / "qualification" / hardware))}', '']
        for system, command in servers.items():
            lines += [f'[servers.{system}]', 'host = "127.0.0.1"', 'port = 18083', f'command = {json.dumps(command)}']
            if system.startswith('fastvideo'):
                lines += [f'[servers.{system}.environment]', 'FASTVIDEO_ATTENTION_BACKEND = "VIDEO_SPARSE_ATTN_H3"', 'FASTVIDEO_VSA_SM100A = "1"', 'FASTVIDEO_FA4 = "1"', 'FASTVIDEO_MINIMAX_H3_FUSIONS = "all"']
            elif system.startswith('vllm'):
                lines += [f'[servers.{system}.environment]', 'VLLM_WORKER_MULTIPROC_METHOD = "spawn"', 'VLLM_OMNI_VIDEO_SYNC_TIMEOUT = "21600"']
                if hardware.startswith('gb200'):
                    lines += ['FASTVIDEO_VSA_SM100A = "1"']
            elif system.startswith('sglang'):
                lines += [f'[servers.{system}.environment]', 'PYTHONPATH = "/workspace/UniServe/refs/sglang-current/python"', 'EP_SUPPRESS_NCCL_CHECK = "1"', 'FASTVIDEO_VSA_SM100A = "1"']
            lines += ['']
        names = []
        for system in servers:
            backend, tier = system.rsplit('-', 1)
            points = [('qualification', 1, 0, 6), ('latency', 1, 0, 72)]
            points += [('throughput', concurrency, repetition, 32) for concurrency in ((1, 2, 4, 8) if gpus == 4 else (1, 2, 4, 8, 16)) for repetition in (1, 2, 3)]
            for scenario, concurrency, repetition, count in points:
                name = f'{system}-{scenario}-c{concurrency}' + (f'-r{repetition}' if repetition else '')
                names.append(name)
                manifest = f'throughput-r{repetition}' if repetition else scenario
                endpoint = '/v1/videos' if backend == 'sglang' else '/v1/videos/sync'
                lines += [f'[benchmarks.{name}]', f'server = "{system}"', 'task = "video"', 'model = "FastH3"', 'dataset = "jsonl"', f'dataset_path = {json.dumps(str(PROTOCOL / (manifest + ".jsonl")))}', f'endpoint = "{endpoint}"', f'[benchmarks.{name}.load]', f'num_prompts = {count}', 'request_rate = inf', f'max_concurrency = {concurrency}', 'warmup_requests = 0', 'request_timeout_s = 21600', 'seed = 20260925', f'warmup_manifest = {json.dumps(str(PROTOCOL / "warmup.jsonl"))}']
                if repetition:
                    lines += [f'priming_manifest = {json.dumps(str(PROTOCOL / f"priming-c{concurrency}-r{repetition}.jsonl"))}']
                lines += [f'[benchmarks.{name}.video]', f'backend = "{backend}"', 'poll_interval_s = 0.1']
                if scenario != 'qualification':
                    lines += [f'media_dir = {json.dumps(str(CAMPAIGN / "media" / hardware / tier / backend / scenario / name))}']
                lines += [f'[benchmarks.{name}.metrics]', 'videos_per_second = "higher"', '"video_latency_ms.p95" = "lower"', '']
                lines += [f'[benchmarks.{name}.provenance]', f'hardware = "{hardware}"', f'checkpoint_revision = "{lock["checkpoints"][tier]["revision"]}"', f'checkpoint = "{MODEL + ("-NVFP4" if tier == "nvfp4" else "")}"', f'system_revision = "{lock["sources"][backend]["revision"]}"', f'source_lock = {json.dumps(str(CAMPAIGN / "protocol/workload/source-lock.json"))}', f'deployment = "{deployment if backend == "uniserve" else system}"', '']
        lines += ['[suites.all]', f'points = {json.dumps(names)}']
        (PROFILES / f'{hardware}.toml').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
