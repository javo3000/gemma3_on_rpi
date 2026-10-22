#!/usr/bin/env bash
# ==============================================================================
# run_inference.sh: Calibrated Execution Wrapper for Gemma 3 on Raspberry Pi
# ==============================================================================

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/gemma-3-4b-it-Q4_K_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-$HOME/models/mmproj-gemma-3-4b-f16.gguf}"
IMAGE_PATH="${1:-}"
OUTPUT_PATH="${2:-$HOME/predictions/prediction_output.txt}"
PROMPT="${PROMPT:-Transcribe all handwritten clinical text from this document image accurately. Output only the transcription.}"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: $0 <path_to_document_image> [output_file_path]"
    echo "Example: $0 ~/datasets/AMR_001_HTR.png ~/predictions/AMR_001.txt"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MMPROJ_PATH" ]; then
    echo "Error: Vision projector file not found at $MMPROJ_PATH"
    exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found at $IMAGE_PATH"
    exit 1
fi

echo "======================================================================="
echo "Running Gemma 3 4B Multimodal OCR on Edge..."
echo "Model:       $MODEL_PATH"
echo "Projector:   $MMPROJ_PATH"
echo "Image:       $IMAGE_PATH"
echo "Output:      $OUTPUT_PATH"
echo "Constraints: Context: 1024 | Max Tokens: 128 | Threads: 4"
echo "======================================================================="

# Execute with memory & thread bounds to prevent Linux kernel OOM termination
"$HOME/llama.cpp/build/bin/llama-mtmd-cli" \
    -m "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --image "$IMAGE_PATH" \
    -p "$PROMPT" \
    -n 128 \
    -c 1024 \
    -t 4 | tee "$OUTPUT_PATH"

echo ""
echo "Transcription successfully saved to: $OUTPUT_PATH"
