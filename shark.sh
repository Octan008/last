# nohup python train.py --config configs/shark.txt &
export CUDA_VISIBLE_DEVICES="1"
python train_recon.py --config configs/base
# python train.py --config configs/shark.txt --ckpt ./log/tensorf_shark_VM/tensorf_shark_VM.th --render_only 1 --render_test 1 

# python train.py --config configs/shark.txt --render_only 1 --render_test 1 
# python train.py --config configs/shark.txt --render_only 1 --render_test 0
# python train.py --config configs/shark.txt # --render_only 1 --render_test 1
# python train.py --config configs/dist.txt # --render_only 1 --render_test 1
# python train_recon.py --config configs/shark_gtBwf_tvloss.txt # --render_only 1 --render_test 1
