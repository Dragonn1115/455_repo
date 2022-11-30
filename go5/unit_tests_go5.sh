#!/bin/bash

# Script for running unit tests for go5

TESTS="test_board.py test_board_util.py"
# TODO test the go5-specific parts
 
for unit_test in $TESTS; do
    python3 $unit_test
done
