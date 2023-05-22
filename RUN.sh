# DIP Final Project

if [ ! -d "./result" ]
then
   mkdir ./result 
fi

if [ ! -d "./result/test" ]
then
   mkdir ./result/test
fi

python3 ImageInpainting.py --input ./dataset/data8/image.png --mask ./dataset/data8/mask.png --output ./result/result8.png --patch_size 11
