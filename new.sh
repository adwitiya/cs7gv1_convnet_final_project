NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"

(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.1 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.001 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.03 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.3 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 1 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.0001 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.05 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.003 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.00001 --cuda --datapath ./data --model adwitiya --optim Sgd --aug SCALE_H_FLIP --tag Testingtest --epochs 24 --batch_size 64) | (tee "$LOGFILE")
