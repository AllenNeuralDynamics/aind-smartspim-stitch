#!/bin/bash

# -------------------------------------------------------------
# This script configures a Google Cloud Run Job for the stitching pipeline. 
#
# The parameters are:
# (str): Google project where the Cloud Run Job will be deployed.
# (str): Google run region.
#
# Example:
#
# -------------------------------------------------------------

if [ -z "$1" ]
    then
        echo "Plase, introduce the Google Project."
        exit 1
fi

if [ -z "$2" ]
    then
        echo "Using default region (us-west1)"
        region="us-west1"
else
    region=$2
fi

# Setting project
gcloud config set project $1 && \
# Setting run region
gcloud config set run/region $region && \
# Build image
gcloud builds submit --tag us.gcr.io/$1/stitch-images/terastitcher && \
# Creating job
gcloud beta run jobs create terastitcher --image us.gcr.io/$1/stitch-images/terastitcher --cpu 8 --memory 32G

# Update and execute jobs with new dataset
# gcloud beta run jobs update terastitcher --args=--input_data,gs://aind-data-dev/camilo.laiton/unstitched,--output_data,gs://aind-data-dev/camilo.laiton/stitched_cloud_run

# Execute job
# gcloud beta run jobs execute terastitcher

python3 -m \
    apache_beam.examples.wordcount \
    --region us-central1 --input \
    gs://dataflow-samples/shakespeare/kinglear.txt \
    --output \
    gs://aind-data-dev/camilo.laiton/dataflow_test \
    --runner DataflowRunner \
    --project neural-dynamics-dev \
    --temp_location \
    gs://aind-data-dev/camilo.laiton/dataflow_test/temp/