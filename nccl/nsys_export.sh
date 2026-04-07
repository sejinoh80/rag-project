#!/bin/bash

set -ex

NSYS_REP=$1
BASE_NAME=${NSYS_REP%.*}
echo "NSYS_REP: $NSYS_REP"
echo "BASE_NAME: $BASE_NAME"

nsys export \
    --type sqlite \
    --output $BASE_NAME.sqlite \
    $NSYS_REP