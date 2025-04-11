output "service_url" {
  value       = google_cloud_run_v2_service.service.uri
  description = "Cloud Runサービスのデプロイされたエンドポイント"
}

output "service_name" {
  value       = google_cloud_run_v2_service.service.name
  description = "Cloud Runサービス名"
}

output "service_id" {
  value       = google_cloud_run_v2_service.service.id
  description = "Cloud RunサービスのID"
}
