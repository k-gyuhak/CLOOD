python main.py \
--architecture deit_small_patch16_224 \
--transformer adapter_hat \
--method ROW \
--dataset timgnet \
--n_tasks 10 \
--adapter_latent 128 \
--optimizer sgd \
--compute_md \
--compute_auc \
--use_md \
--buffer_size 2000 \
--n_epochs 10 \
--lr 0.005 \
--use_buffer \
--class_order 0 \
--folder timgnet_10t