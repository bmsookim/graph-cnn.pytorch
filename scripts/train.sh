python train.py \
    --dataset pubmed \
    --num_hidden 128 \
    --dropout 0.5 \
    --weight_decay 0 \
    --model basic \
    --lr 1e-2 \
    --optimizer SGD \
    --epoch 10000 \
    --lr_decay_epoch 2500
