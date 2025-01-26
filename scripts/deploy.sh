#!/bin/bash

set -e

# Build Docker image
docker build -t customer-insight-api:latest .

# Push to container registry
docker push your-registry.com/customer-insight-api:latest

# Update Kubernetes deployment
kubectl set image deployment/customer-insight-api customer-insight-api=your-registry.com/customer-insight-api:latest

# Wait for rollout to complete
kubectl rollout status deployment/customer-insight-api

echo "Deployment completed successfully"
