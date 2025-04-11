resource "google_cloudbuild_trigger" "github_trigger" {
  name        = var.trigger_name
  description = var.description
  project     = var.project_id
  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = var.branch_regex
    }
  }
  service_account = var.service_account_id
  filename = "cloudbuild.yaml"

  substitutions = {
    _SERVICE_NAME        = var.service_name
    _GOOGLE_CLOUD_REGION = var.repository_region
  }


}