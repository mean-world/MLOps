# install kubeflow pipeline & CRD
```
export PIPELINE_VERSION=2.4.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

# install ray cluster
```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
# Install both CRDs and KubeRay operator v1.3.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0 --namespace ray
helm install raycluster kuberay/ray-cluster --version 1.3.0 -f kuberay/value.yaml --namespace ray
```

# install mlflow server
```
kubectl apply -f mlflow/mlflow.yaml
```

# install gitlab server & runner
```
kubectl apply -f gitlab/gitlab.yaml 
#get runner token paste to gitlab_runner.yaml ConfigMap token 
kubectl apply -f gitlab_runner.yaml
```