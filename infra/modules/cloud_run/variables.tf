variable "project_id" {
  description = "GCPプロジェクトID"
  type        = string
}
variable "description" {
  description = "Cloud Runサービスの説明"
  type        = string
}

variable "region" {
  description = "デプロイするリージョン"
  type        = string
}

variable "service_name" {
  description = "Cloud Runサービス名"
  type        = string
}

variable "image" {
  description = "コンテナイメージのURL"
  type        = string
}

variable "min_instances" {
  description = "最小インスタンス数"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "最大インスタンス数"
  type        = number
  default     = 4
}

variable "memory" {
  description = "メモリ割り当て"
  type        = string
  default     = "512Mi"
}

variable "cpu" {
  description = "CPU割り当て"
  type        = string
  default     = "1"
}

variable "num_gpus" {
  description = "GPU数"
  type        = number
  default     = 0
}

variable "startup_cpu_boost" {
  description = "起動時のCPUブースト"
  type        = bool
  default     = false
}

variable "accelerator" {
  description = "gpuの種類"
  type        = string
  default     = "nvidia-l4"
}

variable "timeout_seconds" {
  description = "リクエストタイムアウト（秒）"
  type        = number
  default     = 300
}

variable "env_vars" {
  description = "環境変数のマップ"
  type        = map(string)
  default     = {}
}

variable "port" {
  description = "ポート番号"
  type        = number
  default     = 8080
}

variable "short_sha" {
  description = "デプロイするコミットのSHA"
  type        = string
  default     = ""
}

variable "secrets" {
  description = "Secret Managerから取得する環境変数のマップ"
  type = map(object({
    secret_id = string
    version   = string
  }))
  default = {}
}
