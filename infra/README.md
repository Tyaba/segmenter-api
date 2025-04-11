# terraform infrastructure of segmenter API

# Prerequisites
## create bucket
```bash
$ bash infra/bin/setup.sh
```
## CI/CD setup
GitHubリポジトリをCloud Buildと接続 (Repository mapping)
（CloudBuild > トリガー > リポジトリを接続）
<img width="1725" alt="Image" src="https://github.com/user-attachments/assets/3b04a9df-0ec9-418b-bae6-3994c4afc438" />

# How to Use
```bash
$ cd infra/envs/dev
$ terraform init
$ terraform plan # check
$ terraform apply
```