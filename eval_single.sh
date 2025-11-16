python comprehensive_evaluation.py \
    --model checkpoints/best_ms_hrnet_v2_dat_4bands.pth \
    --model-type ms_hrnet_v2 \
    --test-img /mnt/U/Dat_Seg/dat_4bands/val/images/ \
    --test-mask /mnt/U/Dat_Seg/dat_4bands/val/labels/ \
    --output-dir evaluation_results/ms_hrnet_v2 \
    --visualize