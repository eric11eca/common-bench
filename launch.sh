#!/bin/bash

CLUSTER_USER=zechen 			# find this by running `id -un` on iccvlabsrv
CLUSTER_USER_ID=254670  		# find this by running `id -u` on iccvlabsrv
CLUSTER_GROUP_NAME=NLP-StaffU   # find this by running `id -gn` on iccvlabsrv
CLUSTER_GROUP_ID=11131 			# find this by running `id -g` on iccvlabsrv

MY_IMAGE="ic-registry.epfl.ch/nlp/common_bench_2"

# arg_job_suffix=$1
# arg_job_name="$CLUSTER_USER-inter$arg_job_suffix"

arg_job_name="macaw1"

echo "Job [$arg_job_name]"

runai submit $arg_job_name \
	-i $MY_IMAGE \
	--interactive \
	--attach 
	--gpu 1 \
    --cpu 4 \
    -e CLUSTER_USER=$CLUSTER_USER \
	-e CLUSTER_USER_ID=$CLUSTER_USER_ID \
	-e CLUSTER_GROUP_NAME=$CLUSTER_GROUP_NAME \
	-e CLUSTER_GROUP_ID=$CLUSTER_GROUP_ID \
    --pvc runai-nlp-zechen-nlpdata1:/nlpdata1 \
	--large-shm \
    --service-type=nodeport \
    --port 30014:22


# check if succeeded
if [ $? -eq 0 ]; then
	runai describe job $arg_job_name

# 	echo ""
# 	echo "Connect - terminal:"
# 	echo "	runai bash $arg_job_name"
fi
