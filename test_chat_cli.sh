#!/bin/bash
set -e

set -x

TIMESTAMP=$(date +%s)
USERNAME="user_$TIMESTAMP"
EMAIL="user_$TIMESTAMP@test.com"
PASSWORD="password123"

echo "Testing with User: $USERNAME"

# Run chat-cli with input
./chat-cli <<EOF
2
$USERNAME
$EMAIL
$PASSWORD
1
$USERNAME
$PASSWORD
1
Test Chat
Hello, this is a test message.
exit
5
EOF
