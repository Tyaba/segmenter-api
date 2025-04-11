output "email" {
  value       = google_service_account.service_account.email
  description = "サービスアカウントのメールアドレス"
}

output "id" {
  value       = google_service_account.service_account.id
  description = "サービスアカウントのID"
}

output "name" {
  value       = google_service_account.service_account.name
  description = "サービスアカウントの完全な名前"
}
