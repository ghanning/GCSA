#!/bin/bash
set -e

script_dir=`dirname $0`
pushd $script_dir/..

for scene in CAB HGE LIN
do
    python -m hloc.extract_features --image_dir datasets/lamar/$scene --export_dir outputs/lamar/$scene --conf netvlad --as_half
done

popd
