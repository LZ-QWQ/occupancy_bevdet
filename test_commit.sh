CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash ./tools/dist_test.sh ./configs/bevdet_occ/TTA_InternImage_672x1408_test.py \
 ./ckpts/bevdet-InternImage-672*1408-epoch_4_ema.pth 8 \
 --format-only --eval-options 'submission_prefix=./results/672x1408_TTA/val46.08_TTA'

# ./tools/dist_test.sh configs/bevdet_occ/test_internimage.py \
# work_dirs/bevdet-occ-intenimage_B_custom_decay-4d-stereo-512x1408-24e-labelsmooth_0.00001-load_cbgs/epoch_9_ema.pth \
# 8 --format-only --eval-options 'submission_prefix=./results/merge_logit/internimage_val45.52_TTA'
