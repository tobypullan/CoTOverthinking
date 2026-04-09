#!/bin/bash

set -euo pipefail

HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
QWEN_CACHE_DIR="${HF_HOME_DIR}/hub/models--Qwen--Qwen3-32B"
LOCK_DIR="${HF_HOME_DIR}/hub/.locks/models--Qwen--Qwen3-32B"
REMOVE_XET_CACHE=false

if [[ "${1:-}" == "--remove-xet-cache" ]]; then
  REMOVE_XET_CACHE=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--remove-xet-cache]"
  exit 1
fi

echo "HF_HOME: $HF_HOME_DIR"

if [[ -d "$QWEN_CACHE_DIR" ]]; then
  echo "Removing Qwen model cache:"
  echo "  $QWEN_CACHE_DIR"
  du -sh "$QWEN_CACHE_DIR" || true
  rm -rf "$QWEN_CACHE_DIR"
else
  echo "Qwen model cache not found:"
  echo "  $QWEN_CACHE_DIR"
fi

if [[ -d "$LOCK_DIR" ]]; then
  rm -rf "$LOCK_DIR"
fi

if [[ "$REMOVE_XET_CACHE" == true ]]; then
  XET_DIR="${HF_HOME_DIR}/xet"
  if [[ -d "$XET_DIR" ]]; then
    echo "Removing the full Hugging Face xet cache:"
    echo "  $XET_DIR"
    du -sh "$XET_DIR" || true
    rm -rf "$XET_DIR"
  fi
fi

echo "Remaining Hugging Face cache usage:"
du -sh "$HF_HOME_DIR" 2>/dev/null || true
