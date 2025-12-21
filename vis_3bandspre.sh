python select_best_visualizations.py \
    --test-img /home/lucianlu/data/dat_4bands/val/images/ \
    --test-mask /home/lucianlu/data/dat_4bands/val/labels/ \
    --models-config eval_3bands.json \
    --in-ch 3 \
    --strategy best \
    --num-samples 10 \
    --output-dir fig/3bands_pre \