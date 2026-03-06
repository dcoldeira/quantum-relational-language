#!/bin/bash
# Usage: ./commit.sh "feat: description"
# Commits staged changes and pushes to origin main.

msg=$1

if [ -z "$msg" ]; then
  echo "Usage: ./commit.sh 'message'"
  exit 1
fi

git commit -m "$msg" && git push origin main
