# The repository includes the main codes for "Low-bit Quantization for Deep Graph Neural Networks with Smoothness-aware Message Propagation"

### Envrioment requirement:

Python: 2.8.2

Pytorch: 1.7.0+cu110

pytorch-geometric: torch-1.7.0+cu112


### Examples:
 
#### Node classification tasks:
>FP32 case:

! python3 main.py --device cpu --qtype FP32  --dataset Cora --lambda_threshold 0.00001 --alpha_threshold 0.1 --random_splits 2 --runs 1 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.8 --K 10 --lambda1 9 --lambda2 9  --prop_mode SMP --epochs 200 

! python3 main.py --device cpu --qtype FP32  --dataset Cora --random_splits 2 --runs 1 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.8 --K 10 --lambda1 9 --lambda2 9  --prop_mode EMP --epochs 200 

>INT8 case:

! python3 main.py --device cpu --lambda_threshold 0.00001 --alpha_threshold 0.00001 --qtype INT8 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00001 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.8 --K 10 --lambda1 9 --lambda2 9  --prop_mode SMP --epochs 200 

! python3 main.py --device cpu --qtype INT8 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00005 --lr1 0.01 --weight_decay1 0.00005 --dropout 0.8 --K 10 --lambda1 9 --lambda2 9  --prop_mode EMP --epochs 200 

>INT4 case:

! python3 main.py --device cpu --lambda_threshold 0.00001 --alpha_threshold 0.00001 --qtype INT4  --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00005 --lr1 0.015 --weight_decay1 0.00001 --dropout 0.5 --K 10 --lambda1 9 --lambda2 9  --prop_mode SMP --epochs 200

! python3 main.py --device cpu --qtype INT4 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00005 --lr1 0.015 --weight_decay1 0.00001 --dropout 0.5 --K 10 --lambda1 9 --lambda2 9  --prop_mode EMP --epochs 200

>INT2 case:

! python3 main.py --device cpu --lambda_threshold 0.00001 --alpha_threshold 0.0001 --qtype INT2 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00001 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.5 --K 10 --lambda1 9 --lambda2 9  --prop_mode SMP --epochs 200 

! python3 main.py --device cpu --qtype INT2 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00001 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.5 --K 10 --lambda1 9 --lambda2 9  --prop_mode EMP --epochs 200 


>INT2-BT(INT2-BT^{*}):

! python3 main.py --device cpu --lambda_threshold 0.00001 --alpha_threshold 0.2 --qtype INT2 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00001 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.5 --K 10 --lambda1 9 --lambda2 9  --prop_mode SMP --epochs 200 --BT_mode BT

! python3 main.py --device cpu --lambda_threshold 0.00001 --alpha_threshold 0.1 --qtype INT2 --dataset Cora --random_splits 2 --runs 1 --lr 0.0001 --weight_decay 0.00001 --lr1 0.01 --weight_decay1 0.00001 --dropout 0.5 --K 10 --lambda1 9 --lambda2 9  --prop_mode SMP --epochs 200  --BT_mode SBT

# Citation
@inproceedings{wang2023low,

  title={Low-bit Quantization for Deep Graph Neural Networks with Smoothness-aware Message Propagation},
  
  author={Wang, Shuang and Eravci, Bahaeddin and Guliyev, Rustam and Ferhatosmanoglu, Hakan},
  
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  
  pages={2626--2636},
  
  year={2023}
}
