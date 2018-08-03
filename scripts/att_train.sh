python train.py \
    --dataset cora \
    --num_hidden 8 \
    --nb_heads 8 \
    --dropout 0.6 \
    --weight_decay 5e-4 \
    --model res_attention \
    --lr 5e-3 \
    --optimizer adam \
    --epoch 800 
