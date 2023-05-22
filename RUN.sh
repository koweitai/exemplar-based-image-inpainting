# DIP Final Project

if [ ! -d "./result" ]
then
   mkdir ./result 
fi

if [ ! -d "./result/test" ]
then
   mkdir ./result/test
fi

python3 ImageInpainting.py --input ./dataset/data9/image.png --mask ./dataset/data9/mask.png --output ./result/result9.png --patch_size 11
