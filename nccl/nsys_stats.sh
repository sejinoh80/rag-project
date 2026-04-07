#!/bin/bash

set -ex

NSYS_SQLITE=$1

echo "NSYS_SQLITE: $NSYS_SQLITE"

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output ./nsys_stats \
    $NSYS_SQLITE
