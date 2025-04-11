output "trigger_id" {
  value       = google_cloudbuild_trigger.github_trigger.id
  description = "作成されたCloud BuildトリガーのID"
}

output "trigger_name" {
  value       = google_cloudbuild_trigger.github_trigger.name
  description = "Cloud Buildトリガー名"
}