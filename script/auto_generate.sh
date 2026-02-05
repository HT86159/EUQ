#!/bin/bash
while true; do
    echo "start..."
    test_model=Qwen2.5
    dataset=hallucination
    HF_ENDPOINT=https://hf-mirror.com
    image_path='../datasets/hallucination'

    python ../generate_answers.py \
        --model_name "$test_model" \
        --dataset "$dataset" \
        --image_path "$image_path"
    echo "finish"
    exit 0
done
