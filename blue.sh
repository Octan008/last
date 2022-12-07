# nohup python train.py --config configs/shark.txt &
# python train.py --config configs/shark.txt
# python train.py --config configs/shark.txt --ckpt ./log/tensorf_shark_VM/tensorf_shark_VM.th --render_only 1 --render_test 1 

# python train.py --config configs/shark.txt --render_only 1 --render_test 1 
# python train.py --config configs/shark.txt --render_only 1 --render_test 0
# python train.py --config configs/shark.txt # --render_only 1 --render_test 1
# python train.py --config configs/dist.txt # --render_only 1 --render_test 1
python train.py --config configs/shark_gtsh.txt --rank_criteria 0
