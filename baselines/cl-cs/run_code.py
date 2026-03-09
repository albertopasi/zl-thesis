import subprocess

# THU-EP: 10-fold cross-subject pretraining (folds 0-9)
# Run one fold per GPU; adjust --gpu-index as available

commands = [
    'python main_pretrain.py --training-fold 0 --gpu-index 0 --cls 9 >0.out',
    'python main_pretrain.py --training-fold 1 --gpu-index 1 --cls 9 >1.out',
    'python main_pretrain.py --training-fold 2 --gpu-index 2 --cls 9 >2.out',
    'python main_pretrain.py --training-fold 3 --gpu-index 3 --cls 9 >3.out',
    'python main_pretrain.py --training-fold 4 --gpu-index 4 --cls 9 >4.out',
    'python main_pretrain.py --training-fold 5 --gpu-index 5 --cls 9 >5.out',
    'python main_pretrain.py --training-fold 6 --gpu-index 6 --cls 9 >6.out',
    'python main_pretrain.py --training-fold 7 --gpu-index 7 --cls 9 >7.out',
    'python main_pretrain.py --training-fold 8 --gpu-index 0 --cls 9 >8.out',
    'python main_pretrain.py --training-fold 9 --gpu-index 1 --cls 9 >9.out',
]

processes = []
for command in commands:
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
    print('Run the command:', command)

for process in processes:
    process.wait()
