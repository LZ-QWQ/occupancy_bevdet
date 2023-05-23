# CUDA_VISIBLE_DEVICES="4,5,6,7" \
# bash ./tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-intenimage_B-4d-stereo-512x1408-24e-labelsmooth.py ckpts/mask_rcnn_internimage_b_fpn_3x_coco.pth 4 --eval miao

CUDA_VISIBLE_DEVICES="4,5,6,7" \
bash ./tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408-24e_labelsmooth.py \
./work_dirs/bevdet-occ-stbase-4d-stereo-512x1408-24e_labelsmooth/epoch_11_ema.pth 4 --eval miao