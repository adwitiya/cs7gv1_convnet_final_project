

#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.001 --cuda --pretrained --datapath ./data --model ResNet18 --aug SCALE_H_FLIP --tag Adwitiya --epochs 25 --batch_size 64) | (tee "$LOGFILE")

#NOW=$(date +"%m_%d_%Y %H:%M:%S")
#LOGFILE="logs/$NOW.log"

#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.001 --cuda --datapath ./data --model ResNet18 --aug SCALE_H_FLIP --tag Adwitiya --epochs 25 --batch_size 64) | (tee "$LOGFILE")

#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --datapath ./data --model Demo --aug H_FLIP --tag Adwitiya --epochs 25 --batch_size 64) | (tee "$LOGFILE")

#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --datapath ./data --model AlexNet --aug SCALE_H_FLIP --tag Adwitiya --epochs 25 --batch_size 128) | (tee "$LOGFILE")
#NOW=$(date +"%m_%d_%Y %H:%M:%S")
#LOGFILE="logs/$NOW.log"
#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --datapath ./data --model AlexNet --aug SCALE_H_FLIP --tag Adwitiya --epochs 25 --batch_size 128) | (tee "$LOGFILE")

#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --datapath ./data --model adwitiya --aug SCALE_H_FLIP --tag AdwitiyaModel --epochs 25 --batch_size 128) | (tee "$LOGFILE")

#NOW=$(date +"%m_%d_%Y %H:%M:%S")
#LOGFILE="logs/$NOW.log"
#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --datapath ./data --model VGG13 --aug SCALE_H_FLIP --tag AdwitiyaModel --epochs 25 --batch_size 32) | (tee "$LOGFILE")

#NOW=$(date +"%m_%d_%Y %H:%M:%S")
#LOGFILE="logs/$NOW.log"
#(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.01 --cuda --pretrained --datapath ./data --model VGG13 --aug SCALE_H_FLIP --tag AdwitiyaModel --epochs 25 --batch_size 32) | (tee "$LOGFILE")


NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.001 --optim Sgd --cuda --pretrained --datapath ./data --model VGG13 --aug SCALE_H_FLIP --tag AdwitiyaVGG --epochs 25 --batch_size 32) | (tee "$LOGFILE")

NOW=$(date +"%m_%d_%Y %H:%M:%S")
LOGFILE="logs/$NOW.log"		
(python -u /users/pgrad/chakraad/cs7gv4_adwitiya/pytorch-scripts/Example.py --lr 0.001 --optim Sgd --cuda --datapath ./data --model VGG13 --aug SCALE_H_FLIP --tag AdwitiyaVGG --epochs 25 --batch_size 32) | (tee "$LOGFILE")
																																																																						
