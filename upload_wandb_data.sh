DATA_PATH=$1
DATA_NAME=$2

WANDB_API_KEY="9edee5b624841e10c88fcf161d8dc54d4efbee29"

if [ "${WANDB_API_KEY}" == "" ]; then
    exit "please specify wandb api key"
fi
if [ "${DATA_PATH}" == "" ]; then
    exit "please specify target directory"
fi
if [ "${DATA_NAME}" == "" ]; then
    exit "please specify dataset name"
fi

export WANDB_ENTITY="epfl_nlp_phd"
export WANDB_PROJECT="data-collection"

wandb artifact put --name ${2} ${1}
