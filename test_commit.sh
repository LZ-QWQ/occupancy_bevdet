# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
# bash ./tools/dist_test.sh ./configs/bevdet_occ/TTA_InternImage_672x1408_test.py \
#  ./ckpts/bevdet-InternImage-672*1408-epoch_9_ema.pth 8 \
#  --format-only --eval-options 'submission_prefix=./results/merge_logit/debug_tta'

bash ./tools/dist_test.sh \
 configs/bevdet_occ/bevdet-occ-intenimage_B_custom_decay-4d-stereo-672x1408-24e-labelsmooth_0.0001-load-CBGS.py \
 ./ckpts/bevdet-InternImage-672*1408-epoch_9_ema.pth \
 8 --format-only --eval-options 'submission_prefix=./results/merge_logit/debug_simple'