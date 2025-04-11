terraform {
  backend "gcs" {
    bucket = "segmenter-api"
    prefix = "terraform/dev/state"
  }
}