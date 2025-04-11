variable "region" {
  description = "リージョン"
  type        = string
}

variable "service_name" {
  description = "サービス名"
  type        = string
}

variable "project_id" {
  description = "プロジェクトID"
  type        = string
}

variable "github_owner" {
  description = "GitHubのリポジトリの所有者"
  type        = string
  default     = "Tyaba"
}

variable "github_repo" {
  description = "GitHubのリポジトリ名"
  type        = string
  default     = "segmenter-api"
}

variable "short_sha" {
  description = "デプロイするコミットのSHA"
  type        = string
  default     = ""
}

variable "branch_regex" {
  description = "GitHubのリポジトリのブランチ名"
  type        = string
  default     = "^main$"
}
