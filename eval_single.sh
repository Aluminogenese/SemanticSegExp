python comprehensive_evaluation.py \
    --model checkpoints/BEST_ms_hrnet_dat_4bands.pth \
    --model-type ms_hrnet \
    --test-img /mnt/U/Dat_Seg/dat_4bands/test/images/ \
    --test-mask /mnt/U/Dat_Seg/dat_4bands/test/labels/ \
    --output-dir evaluation_results/ms_hrnet \
    --visualize