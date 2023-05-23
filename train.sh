# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
# bash ./tools/dist_train.sh ./configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e_labelsmooth.py 8

# CUDA_VISIBLE_DEVICES="0,3" \
# bash ./tools/dist_train.sh ./configs/bevdet_occ/bevdet-occ-intenimage_s-4d-stereo-512x1408-24e-labelsmooth.py 2

# CUDA_VISIBLE_DEVICES="0,3" \
# bash ./tools/dist_train.sh ./configs/bevdet_occ/bevdet-occ-intenimage_B-4d-stereo-512x1408-24e-labelsmooth.py 2

CUDA_VISIBLE_DEVICES="4, 5, 6, 7" \
bash ./tools/dist_train.sh ./configs/bevdet_occ/bevdet-occ-intenimage_B-4d-stereo-512x1408-24e-labelsmooth_0.0001.py 4