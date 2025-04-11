output "repository_id" {
  description = "リポジトリのID"
  value       = google_artifact_registry_repository.docker.repository_id
}

output "repository_name" {
  description = "リポジトリの完全な名前"
  value       = google_artifact_registry_repository.docker.name
}

output "repository_url" {
  description = "リポジトリのURL"
  value       = google_artifact_registry_repository.docker.name
}