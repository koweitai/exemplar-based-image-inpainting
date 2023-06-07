# DIP Final Project

if [ ! -d "./result" ]
then
   mkdir ./result 
fi

python3 ImageInpainting.py --input ./dataset/data1/image.jpg --mask ./dataset/data1/mask.jpg --output ./result/result1.png --patch_size 13
