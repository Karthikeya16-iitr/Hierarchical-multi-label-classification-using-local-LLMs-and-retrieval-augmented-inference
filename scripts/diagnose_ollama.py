#!/usr/bin/env python3
"""Diagnostic script for local Ollama service, installed models, GPU/VRAM, and inference smoke tests."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx

from mavenmagnet.config import load_config
from mavenmagnet.llm_client import OllamaClient, OllamaModelNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("mavenmagnet.diagnose_ollama")


def check_system_resources() -> Dict[str, Any]:
    """Inspect system RAM and GPU availability."""
    info: Dict[str, Any] = {"gpu_detected": False, "system_ram_gb": None, "gpu_details": None}

    # System RAM from /proc/meminfo
    try:
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            with open(meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        info["system_ram_gb"] = round(total_kb / (1024 * 1024), 1)
                        break
    except Exception:
        pass

    # nvidia-smi check
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                info["gpu_detected"] = True
                info["gpu_details"] = res.stdout.strip()
        except Exception:
            pass

    return info


def check_ollama_process_and_vram(base_url: str) -> Dict[str, Any]:
    """Query Ollama /api/ps for models currently loaded in memory and VRAM allocation."""
    result: Dict[str, Any] = {"loaded_models": []}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/ps")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    vram_mb = round(m.get("size_vram", 0) / (1024 * 1024), 1)
                    size_mb = round(m.get("size", 0) / (1024 * 1024), 1)
                    result["loaded_models"].append(
                        {
                            "name": m.get("name"),
                            "size_mb": size_mb,
                            "size_vram_mb": vram_mb,
                        }
                    )
    except Exception:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose local Ollama service and models.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--base-url", type=str, default=None, help="Ollama base URL override")
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Execute inference smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_url = args.base_url or cfg.llm.base_url
    target_model = args.model or cfg.llm.model
    fallback_model = cfg.llm.fallback_model

    print("\n" + "=" * 70)
    print("OLLAMA LOCAL SERVICE & HARDWARE DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. System hardware check
    res_info = check_system_resources()
    print(f"System RAM:        {res_info.get('system_ram_gb', 'Unknown')} GB")
    if res_info.get("gpu_detected"):
        print(f"GPU Info:          {res_info['gpu_details']}")
    else:
        print("GPU Driver CLI:    nvidia-smi not in container PATH (Ollama host service handles GPU offloading)")

    # 2. Check Ollama daemon connectivity
    print(f"Ollama Endpoint:   {base_url}")
    client = OllamaClient(base_url=base_url, model=target_model, timeout_seconds=cfg.llm.timeout_seconds)

    if not client.check_health():
        print("\n[ERROR] Ollama daemon is NOT running or unreachable at " + base_url)
        print("To start Ollama, run: ollama serve")
        sys.exit(1)

    print("Daemon Status:     ONLINE (HTTP 200)")

    # 3. Check loaded models & VRAM
    ps_info = check_ollama_process_and_vram(base_url)
    if ps_info["loaded_models"]:
        print("Active In-Memory Models:")
        for lm in ps_info["loaded_models"]:
            print(f"  - {lm['name']} (Size: {lm['size_mb']} MB, VRAM: {lm['size_vram_mb']} MB)")
    else:
        print("Active Models:     None currently resident in memory (cold start)")

    # 4. Check installed models
    try:
        installed_models = client.list_installed_models()
    except Exception as e:
        print(f"\n[ERROR] Could not list models: {e}")
        sys.exit(1)

    print(f"\nInstalled Models ({len(installed_models)} available):")
    for m in installed_models:
        print(f"  * {m}")

    # 5. Check target candidate model
    print("\n" + "-" * 70)
    print(f"Target Candidate Model: '{target_model}'")

    is_target_installed = client.is_model_installed(target_model)
    if is_target_installed:
        print(f"Status: AVAILABLE and verified on local Ollama service.")
        test_model_to_use = target_model
    else:
        print(f"Status: NOT CURRENTLY INSTALLED.")
        print(f"\n[REQUIRED ACTION]")
        print(f"To make '{target_model}' available, run:")
        print(f"    ollama pull {target_model}")
        print(f"\nAlternatively, configure an installed model in config/config.yaml, e.g.:")
        print(f"    model: \"{fallback_model}\"")

        # Pick fallback model for smoke test if available
        test_model_to_use = fallback_model if client.is_model_installed(fallback_model) else None
        if not test_model_to_use and installed_models:
            test_model_to_use = installed_models[0]

    # 6. Inference smoke test
    if args.smoke_test:
        if not test_model_to_use:
            print(f"\n[SKIPPING SMOKE TEST] No candidate model available to test.")
        else:
            print(f"\nRunning structured JSON smoke test on model: '{test_model_to_use}'...")
            smoke_client = OllamaClient(
                base_url=base_url,
                model=test_model_to_use,
                temperature=0.0,
                timeout_seconds=120.0,
            )
            smoke_prompt = (
                "You are an NLP review classifier. Classify this test review: "
                "\"Deeply cleanses dirt and leaves my skin glowing.\" "
                "Return JSON matching: {\"predictions\": [{\"level_1\": \"Cleansing\", \"level_2\": \"Dirt Removal\"}]}"
            )

            try:
                result = smoke_client.generate(prompt=smoke_prompt, format_json=True)
                print(f"[SUCCESS] Smoke test completed successfully!")
                print(f"  Response:         {result.raw_response.strip()[:150]}")
                print(f"  Inference Time:   {result.duration_seconds:.2f}s")
                print(f"  Generated Tokens: {result.eval_count}")
                print(f"  Generation Speed: {result.tokens_per_second:.1f} tokens/s")
            except Exception as e:
                print(f"[FAILED] Smoke test failed: {e}")
                sys.exit(1)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
