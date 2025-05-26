### Kubeflow Pipeline Project Environment Setup Guide
## Introduction
This document aims to provide the necessary environment setup steps for running our developed Kubeflow Pipeline projects. This includes deploying Kubeflow Pipelines itself, a Ray Cluster, an MLflow Server, and a GitLab Server with its Runner, ensuring all required infrastructure is in place.

# 1. install kubeflow pipeline & CRD
```
export PIPELINE_VERSION=2.4.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

# 2. install ray cluster
```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
# Install both CRDs and KubeRay operator v1.3.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0 --namespace ray
helm install raycluster kuberay/ray-cluster --version 1.3.0 -f kuberay/value.yaml --namespace ray
```

# 3. install mlflow server
```
kubectl apply -f mlflow/mlflow.yaml
```

# 4. install gitlab server & runner
```
kubectl apply -f gitlab/gitlab.yaml 
# After the GitLab server is up and running, you will need to manually configure
# the 'kubeflow_pipline/SimpleUpscale.git' repository within GitLab.
# This typically involves creating a new project and importing the repository,
# or setting up a mirror.

#get runner token paste to gitlab_runner.yaml ConfigMap token 
kubectl apply -f gitlab_runner.yaml
```