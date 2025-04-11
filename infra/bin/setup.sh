# terraform init前に誰かが一度起動すればよい処理
# bucket作成
gcloud storage buckets create gs://${SERVICE_NAME} --location=${GOOGLE_CLOUD_REGION} --project=${GOOGLE_CLOUD_PROJECT}