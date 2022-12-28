export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=1
python train.py --config configs/Blue_mlp_both_elastic --rank_criteria 0
# python train.py --config configs/Blue_debug_mimik --rank_criteria 0
