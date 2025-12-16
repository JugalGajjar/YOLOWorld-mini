#!/bin/bash

# Base command components
PYTHON_SCRIPT="python src/eval_coco.py"
ANN_FILE="--ann_file data/coco/annotations/instances_val2017.json"
IMG_DIR="--img_dir data/coco/val2017/"
OUTPUT_DIR="./coco_eval_logs" # Directory to store output logs

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Colors for better terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "COCO Evaluation - Epochs 0-15"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Loop through epochs 0 to 15
for i in {0..15}; do
    # Create the checkpoint file name
    CHECKPOINT_FILE="checkpoint_epoch_${i}.pth"
    CHECKPOINT_ARG="--checkpoint outputs/${CHECKPOINT_FILE}"

    # Create the log file name
    LOG_FILE="${OUTPUT_DIR}/log_${CHECKPOINT_FILE%.*}.txt"
    
    echo ""
    echo -e "${YELLOW}=========================================="
    echo -e "Epoch $i - $CHECKPOINT_FILE"
    echo -e "==========================================${NC}"
    echo ""

    $PYTHON_SCRIPT \
        $CHECKPOINT_ARG \
        $ANN_FILE \
        $IMG_DIR 2>&1 | tee "$LOG_FILE"
        
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}SUCCESS: Evaluation for $CHECKPOINT_FILE complete${NC}"
        echo -e "${GREEN}   Log saved to: $LOG_FILE${NC}"
    else
        echo -e "${RED}ERROR: Evaluation failed for $CHECKPOINT_FILE${NC}"
        echo -e "${RED}   Check log file for details: $LOG_FILE${NC}"
    fi
    
    echo -e "${YELLOW}----------------------------------------------------------${NC}"
done

echo ""
echo "=========================================="
echo "Script finished!"
echo "=========================================="
echo "All evaluation runs complete."
echo "Logs saved in: $OUTPUT_DIR"
echo "=========================================="