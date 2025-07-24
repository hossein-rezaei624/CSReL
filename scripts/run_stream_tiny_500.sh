export CUBLAS_WORKSPACE_CONFIG=:4096:8

local_path='./results/stream_tiny_imagenet/test6'  # set your output path
dataset='splittinyimagenet'
data_path='./tiny_imagenet_from_DER'  # need to download tiny-imagenet dataset
buffer_size=1000
alpha=1.0
beta=0.0
lr=3e-2
epochs=50
batch_size=32
mem_batch_size=32
use_cuda=1
opt_type='sgd'
ce_factor=1.0
mse_factor=0.0
update_mode='rho_loss'
remove_mode='random'
cur_train_lr=5e-3
cur_train_steps=20
selection_steps=250
setting=''
seed=0


python3 -u stream_continual_learning.py --local_path=$local_path \
	--dataset=$dataset \
	--data_path=$data_path \
	--buffer_size=$buffer_size \
	--alpha=$alpha \
	--beta=$beta \
	--lr=$lr \
	--epochs=$epochs \
	--batch_size=$batch_size \
	--mem_batch_size=$mem_batch_size \
	--use_cuda=$use_cuda \
	--opt_type=$opt_type \
	--ce_factor=$ce_factor \
	--mse_factor=$mse_factor \
	--update_mode=$update_mode \
	--remove_mode=$remove_mode \
	--cur_train_lr=$cur_train_lr \
	--cur_train_steps=$cur_train_steps \
	--selection_steps=$selection_steps \
	--setting=$setting \
	--seed=$seed

