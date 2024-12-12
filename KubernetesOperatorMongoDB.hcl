# Providers
provider "kubernetes" {
  config_path = "~/.kube/config" # Adjust to your kubeconfig path
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

# Variables
variable "namespace" {
  default = "mongodb-operator"
}

# Namespace for MongoDB Operator
resource "kubernetes_namespace" "mongodb_operator" {
  metadata {
    name = var.namespace
  }
}

# Helm Chart for MongoDB Kubernetes Operator
resource "helm_release" "mongodb_operator" {
  name       = "mongodb-kubernetes-operator"
  namespace  = kubernetes_namespace.mongodb_operator.metadata[0].name
  chart      = "mongodb/mongodb-kubernetes-operator"
  repository = "https://mongodb.github.io/helm-charts"
  version    = "0.7.3" # Replace with the desired version

  values = [
    <<EOF
replicaCount: 1
EOF
  ]
}

# MongoDBCluster Custom Resource
resource "kubernetes_manifest" "mongodb_cluster" {
  manifest = {
    apiVersion = "mongodb.com/v1"
    kind       = "MongoDBCommunity"
    metadata = {
      name      = "my-mongodb-cluster"
      namespace = kubernetes_namespace.mongodb_operator.metadata[0].name
    }
    spec = {
      members   = 3
      version   = "6.0.4" # Replace with the desired MongoDB version
      type      = "ReplicaSet"
      persistent = true
      security = {
        authentication = {
          modes = ["SCRAM"]
        }
      }
      statefulSet = {
        spec = {
          volumeClaimTemplates = [
            {
              metadata = {
                name = "data-volume"
              }
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          ]
        }
      }
    }
  }
}
