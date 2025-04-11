variable "project_id" {
  description = "GCPプロジェクトID"
  type        = string
}

variable "account_id" {
  description = "サービスアカウントのID"
  type        = string
}

variable "display_name" {
  description = "サービスアカウントの表示名"
  type        = string
}

variable "description" {
  description = "サービスアカウントの説明"
  type        = string
}

variable "roles" {
  description = "サービスアカウントに付与するロール"
  type        = list(string)
}
