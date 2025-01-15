#!/bin/bash

set -e

# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run end-to-end tests
pytest tests/e2e

# Generate test coverage report
coverage run -m pytest
coverage report -m
coverage html
