#!/bin/bash

if [ -f ./.env ]; then
  set -a
  source ./.env
  set +a
fi
