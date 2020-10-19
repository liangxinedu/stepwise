# Step-wise Deep Learning Models for Solving Routing Problems
Liang Xin, Wen Song, Zhiguang Cao, and Jie Zhang. Step-wise Deep Learning Models for Solving Routing Problems.  IEEE Transactions on Industrial Informatics (TII), 2020. (In Press) [https://ieeexplore.ieee.org/document/9226142]

To download the pretrained models and data
```bash
python download.py;
tar -zxvf data.tar.gz;
cd SWTAM; tar -zxvf SWTAM_pretrained;
cd ../SWTAM; tar -zxvf ASWTAM_pretrained;
cd ..;
```

## Step-wise Transformer Attention Model (SW-TAM)
To train the SW-TAM models for tsp and cvrp, respectively
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20_rollout' --val_dataset ../data/tsp20.pkl
python run.py --graph_size 20 --baseline rollout --run_name 'vrp20_rollout' --val_dataset ../data/vrp20.pkl --problem="cvrp"
 ```
 
To eval with pretrained models for tsp and cvrp, respectively
```bash
python3 run.py --graph_size 20 --load_path pretrained/tsp_20/epoch-99.pt --val_dataset ../data/tsp20.pkl --eval_only
python3 run.py --graph_size 20 --load_path pretrained/cvrp_20/epoch-99.pt --val_dataset ../data/vrp20.pkl --problem="cvrp" --eval_only
```

## Approximate Step-wise Transfomer Attention Model (ASW-TAM)
To train ASW-TAM;
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20_rollout' --val_dataset ../data/tsp20.pkl
```

To eval with pretrained models
```bash
python3 eval.py ../data/tsp20.pkl --model pretrained/tsp_100/epoch-99.pt --decode_strategy greedy
```

## Step-wise Pointer Network (SW-PtrNet)
To train SW-PtrNet
```bash
python -u main.py --hidden_dim=256 --log_step=1000 --checkpoint_secs=30000
```
To eval with pretrained_models
```bash
python -u main.py --hidden_dim=256 --log_step=1000 --checkpoint_secs=30000 --load_path pretrained/tsp_2019-08-19_14-04-45/model.ckpt-18000000 --is_train=False
```

SWTAM and ASWTAM are developed based on https://github.com/wouterkool/attention-learn-to-route
SWPtrNet are developed based on https://github.com/devsisters/pointer-network-tensorflow
