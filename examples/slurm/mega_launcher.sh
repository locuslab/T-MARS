CUDA_VISIBLE_DEVICES=0 nice -n 20 taskset -c 100-109 bash launcher_1.sh 0 &
CUDA_VISIBLE_DEVICES=0 nice -n 20 taskset -c 110-119 bash launcher_1.sh 20 &
CUDA_VISIBLE_DEVICES=0 nice -n 20 taskset -c 80-89 bash launcher_1.sh 30 &
CUDA_VISIBLE_DEVICES=0 nice -n 20 taskset -c 90-99 bash launcher_1.sh 40 &

CUDA_VISIBLE_DEVICES=1 nice -n 20 taskset -c 40-49 bash launcher_1.sh 50 &
CUDA_VISIBLE_DEVICES=1 nice -n 20 taskset -c 50-59 bash launcher_1.sh 60 &
CUDA_VISIBLE_DEVICES=1 nice -n 20 taskset -c 60-69 bash launcher_1.sh 70 &
CUDA_VISIBLE_DEVICES=1 nice -n 20 taskset -c 70-79 bash launcher_1.sh 80