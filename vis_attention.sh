python vis_attention.py \
    --model checkpoints/BEST_ms_hrnet_v2_dat_4bands.pth \
    --image /home/lucianlu/data/dat_4bands/val/images/000000193.tif \
    --output attention_vis \
    --in-ch 4

python vis_attention.py \
    --model checkpoints/BEST_ms_hrnet_v2_dat_4bands.pth \
    --image /home/lucianlu/data/dat_4bands/val/images/000000193.tif \
    --output attention_vis \
    --in-ch 4 \
    --simple