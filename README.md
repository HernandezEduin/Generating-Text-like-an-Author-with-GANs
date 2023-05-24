# Generating Text like an Author with GANs
 Project for Statistical Foundations of Natural Language Processing (EEE486)

## Execution Commands
### char-rnn
```
python train.py "../Dataset/William Shakespeare/shakespeare_train.txt"
python generate.py shakespeare_train.pt --output-filename shakespeare_test_char.txt --cuda --predict_len 400
```
### textGAN
(inside ./run/)
```
	python run_seqgan.py --dataset wells
```
