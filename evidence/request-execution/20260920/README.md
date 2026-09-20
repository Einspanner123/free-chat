# Intermediate failing run

Preserved test output: 474 passed, 12 GPU skips, one new test failed because a protobuf
module import was missing. Ruff/mypy also detected that missing import. This directory
is not acceptance evidence. See sibling `20260920-final/` for the corrected source-pinned
full regression. The failure was in test setup, not waived or marked as a skip.
