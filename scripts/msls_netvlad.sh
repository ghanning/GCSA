#!/bin/bash
set -e

script_dir=`dirname $0`
pushd $script_dir/..

for scene in trondheim london boston melbourne amsterdam helsinki tokyo toronto saopaulo moscow zurich paris bangkok budapest austin berlin ottawa phoenix goa amman nairobi manila cph sf
do
    python -m hloc.extract_features --image_dir datasets/msls/train_val/$scene --export_dir outputs/msls/$scene --conf netvlad --as_half
done

for scene in miami athens buenosaires stockholm bengaluru kampala
do
    python -m hloc.extract_features --image_dir datasets/msls/test/$scene --export_dir outputs/msls/$scene --conf netvlad --as_half
done

popd
