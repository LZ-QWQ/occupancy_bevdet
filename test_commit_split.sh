# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
# bash ./tools/dist_test.sh ./configs/bevdet_occ/TTA_InternImage_672x1408_test.py \
#  ./ckpts/bevdet-InternImage-672*1408-epoch_9_ema.pth 8 \
#  --format-only --eval-options 'submission_prefix=./results/merge_logit/debug_tta'

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash ./tools/TTA_MINI_TEST.sh \
 ./ckpts/bevdet-InternImage-672*1408-epoch_9_ema.pth 8 \
 --format-only --eval-options 'submission_prefix=./results/merge_logit/tta_672x1408_epoch9_ema'