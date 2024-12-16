CUDA_VISIBLE_DEVICES=0 python3 -m compressai.utils.eval_model -d /home/lbuysse/kodak/ -r /home/lbuysse/reconstruction/ -a stf -p /home/lbuysse/saved_models/stf_013_0018_best.pth.tar  --cuda
CUDA_VISIBLE_DEVICES=0 python3 -m compressai.utils.eval_model -d /home/lbuysse/kodak/ -r /home/lbuysse/reconstruction/ -a stf -p /home/lbuysse/saved_models/stf_013_0067_best.pth.tar  --cuda
CUDA_VISIBLE_DEVICES=0 python3 -m compressai.utils.eval_model -d /home/lbuysse/kodak/ -r /home/lbuysse/reconstruction/ -a stf -p /home/lbuysse/saved_models/stf_013_0035_best.pth.tar  --cuda


CUDA_VISIBLE_DEVICES=0 python3 train.py -d /home/lbuysse/dataset/ -e 1 --batch-size 8 --save --checkpoint /home/lbuysse/stf/stf_013.pth.tar --save_path /home/lbuysse/stf/stf_013_0067.pth.tar -m stf --cuda --lambda 0.0067 --adapter