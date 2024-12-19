# run-ragged

```shell

# development packages

devbox shell

# environment variables

export PROJECT_PREFIX='demo'
export PROJECT_NAME='lc-rag'
export PROJECT_SUFFIX=$(date +%Y%m%d)
export PROJECT=pt-${PROJECT_PREFIX}-${PROJECT_NAME}-$PROJECT_SUFFIX

# cloud setup

gcloud auth login
gcloud projects create $PROJECT
gcloud config set project $PROJECT
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT
doppler run --command="gcloud billing projects link \$PROJECT --billing-account=\$GCLOUD_BILLING_FFI"
gcloud config list project

gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  bigqueryconnection.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  cloudresourcemanager.googleapis.com \
  documentai.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com

# add database

doppler run --command="gcloud sql instances create sql-instance \
  --database-version POSTGRES_14 \
  --tier db-f1-micro \
  --region \$GCLOUD_REGION"

# add storage

doppler run --command="gcloud storage buckets create gs://pdf-bucket-\$PROJECT_SUFFIX --project=\$PROJECT --location=\$GCLOUD_REGION"
gcloud sql databases create release-notes --instance sql-instance
doppler run --command="gcloud sql users create \$DB_USERNAME_DEMO --instance sql-instance --password \$DB_PASSWORD_DEMO"

# indexer

doppler run --command="gcloud run jobs deploy indexer \
  --source run-ragged/. \
  --command python \
  --args run-ragged/app/indexer.py \
  --set-env-vars=DB_INSTANCE_NAME=`gcloud sql instances describe sql-instance --format="value(connectionName)"` \
  --set-env-vars=DB_USER=\$DB_USERNAME_DEMO \
  --set-env-vars=DB_NAME=release-notes \
  --set-env-vars=DB_PASS=\$DB_PASSWORD_DEMO \
  --set-env-vars=PDF_BUCKET_NAME=pdf-bucket-$PROJECT_SUFFIX \
  --region=\$GCLOUD_REGION \
  --execute-now"

# web app

doppler run --command="gcloud run deploy run-ragged \
  --source run-ragged/. \
  --set-env-vars=DB_INSTANCE_NAME=`gcloud sql instances describe sql-instance --format="value(connectionName)"` \
  --set-env-vars=DB_USER=\$DB_USERNAME_DEMO \
  --set-env-vars=DB_NAME=release-notes \
  --set-env-vars=DB_PASS=\$DB_PASSWORD_DEMO \
  --set-env-vars=PDF_BUCKET_NAME=pdf-bucket-$PROJECT_SUFFIX \
  --region=\$GCLOUD_REGION \
  --allow-unauthenticated"
```
