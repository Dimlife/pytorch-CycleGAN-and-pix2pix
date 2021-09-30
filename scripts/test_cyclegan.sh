set -ex
python test.py --dataroot ./datasets/maps --name maps_unet_sitt --model sitt --phase test --no_dropout --load_iter 10
