python select_best_visualizations.py \
    --test-img /home/lucianlu/data/dat_4bands/val/images/ \
    --test-mask /home/lucianlu/data/dat_4bands/val/labels/ \
    --models-config eval_4bands.json \
    --in-ch 4 \
    --strategy best \
    --num-samples 10 \
    --output-dir fig/4bands_pre \