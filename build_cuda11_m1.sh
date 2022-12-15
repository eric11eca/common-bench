NUM=3

CURRENT=${NUM}
IMAGE_NAME=common_bench
DOCKERFILE_NAME=Dockerfile

GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE_NAME}_${NUM}

echo "Building $IM_NAME"
docker buildx build --platform linux/amd64 --load -f $DOCKERFILE_NAME -t $IM_NAME --cache-from type=local,src=../../.docker_cache --cache-to type=local,mode=max,dest=../../.docker_cache .

echo "Pushing $IM_NAME to Harbor"
docker tag $IM_NAME ic-registry.epfl.ch/nlp/$IM_NAME
docker push ic-registry.epfl.ch/nlp/$IM_NAME

export KUBECONFIG=~/.kube/config_runai

# runai submit --name common-bench -i ic-registry.epfl.ch/nlp/common_bench_1 --attach --interactive -g 1