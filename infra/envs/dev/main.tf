module "artifact_registry" {
  source = "../../modules/artifact_registry"
  location = var.region
  repository_id = var.service_name
  description = "docker imagesが置かれるリポジトリ"
}

module "builder_service_account" {
  source = "../../modules/iam"
  project_id = var.project_id
  account_id = "${var.service_name}-builder"
  display_name = "segmenter-api-builder"
  description = "segmenter-apiのビルダーサービスアカウント"
  roles = [
    "roles/cloudbuild.builds.builder",
    "roles/resourcemanager.projectIamAdmin",
    "roles/iam.serviceAccountUser",
    "roles/storage.objectUser",  # gcs backendの読み書きのため
    "roles/cloudbuild.builds.editor",
    "roles/logging.logWriter",
    "roles/run.admin",
    "roles/artifactregistry.admin",
  ]
}

module "cloud_run_service_account" {
  source = "../../modules/iam"
  project_id = var.project_id
  account_id = "${var.service_name}-cloud-run"
  display_name = "segmenter-api-cloud-run"
  description = "segmenter-apiのCloud Runサービスアカウント"
  roles = [
    "roles/storage.objectViewer",
  ]
}

module "cloud_build_trigger" {
  source = "../../modules/cloud_build_trigger"
  project_id = var.project_id
  service_account_id = module.builder_service_account.id
  repository_region = var.region
  trigger_name = var.service_name
  service_name = var.service_name
  github_owner = var.github_owner
  github_repo = var.github_repo
  branch_regex = var.branch_regex
  substitutions = {
    _SERVICE_NAME = var.service_name
    _GOOGLE_CLOUD_REGION = var.region
    _ENV = "dev"
  }
  description = "segmenter APIのCloud Buildトリガー"
}

module "segmenter_cloud_run" {
  source = "../../modules/cloud_run"
  project_id = var.project_id
  service_name = var.service_name
  short_sha = var.short_sha
  region = var.region
  image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}:latest"
  num_gpus = 1
  accelerator = "nvidia-l4"
  min_instances = 0
  max_instances = 4
  memory = "32Gi"
  cpu = "8"
  startup_cpu_boost = false
  service_account_email = module.cloud_run_service_account.email
  description = "segmenter APIのCloud Runサービス"
}
