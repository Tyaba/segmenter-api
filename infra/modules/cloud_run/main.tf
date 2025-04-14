resource "google_cloud_run_v2_service" "service" {
  name     = var.service_name
  location = var.region
  project  = var.project_id
  deletion_protection = false
  template {
    service_account = var.service_account_email
    containers {
      image = var.image
      resources {
        limits = {
          "cpu" = var.cpu
          "memory" = var.memory
          "nvidia.com/gpu" = var.num_gpus
        }
        startup_cpu_boost = var.startup_cpu_boost
      }
      dynamic "env" {
        for_each = var.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = var.secrets
        content {
          name = env.key
          value_from {
            secret_key_ref {
              name  = env.value.secret_id
              key = env.value.version
            }
          }
        }
      }
      ports {
        container_port = var.port
      }
    }
    node_selector {
      accelerator = var.accelerator
    }
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    labels = {
      "commit-sha" = var.short_sha != "" ? var.short_sha : "latest"
    }
    gpu_zonal_redundancy_disabled = true
  }
}

