variable "service_name" {
  description = "このprojectのサービス名"
  type        = string
}

variable "project_id" {
  description = "GCPプロジェクトID"
  type        = string
}

variable "trigger_name" {
  description = "Cloud Buildトリガー名"
  type        = string
}

variable "description" {
  description = "トリガーの説明"
  type        = string
}

variable "service_account_id" {
  description = "builderに付与するサービスアカウントID"
  type        = string
}

variable "repository_region" {
  description = "リポジトリのリージョン"
  type        = string
}

variable "github_owner" {
  description = "GitHubリポジトリのオーナー"
  type        = string
}

variable "github_repo" {
  description = "GitHubリポジトリ名"
  type        = string
}

variable "branch_regex" {
  description = "トリガーするブランチの正規表現"
  type        = string
}

variable "substitutions" {
  description = "Substitutions to be used in the build trigger"
  type = map(string)
  default = {}
}
