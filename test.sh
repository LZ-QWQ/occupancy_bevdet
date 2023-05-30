# CUDA_VISIBLE_DEVICES="4,5,6,7" \
# bash ./tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-intenimage_B-4d-stereo-512x1408-24e-labelsmooth.py ckpts/mask_rcnn_internimage_b_fpn_3x_coco.pth 4 --eval miao

# CUDA_VISIBLE_DEVICES="0,1,2,3,4, 5,7" \
# bash ./tools/dist_test.sh ./configs/bevdet_occ/TTA_config_test.py \
# ./work_dirs/bevdet-occ-stbase-4d-stereo-512x1408-24e_labelsmooth/epoch_12_ema.pth 7 --eval miao

# CUDA_VISIBLE_DEVICES="0,1,2,3,4, 5,6,7" \
# bash ./tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-intenimage_B_custom_decay-4d-stereo-672x1408-24e-labelsmooth_0.0001-load-CBGS.py \
#  /data/work_dirs/bevdet-occ-intenimage_B_custom_decay-4d-stereo-672x1408-24e-labelsmooth_0.0001-load-CBGS/epoch_2_ema.pth 8 --eval miao


#  | tee -a ./work_dirs/test_log_internimage_B-LB_0.00001-load.txt


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
bash ./tools/dist_test.sh ./configs/bevdet_occ/bevdet-occ-intenimage_B_custom_decay-4d-stereo-672x1408-24e-labelsmooth_0.0001-load-CBGS.py \
 ./ckpts/bevdet-InternImage-672*1408-epoch_4_ema.pth 8 --eval miao