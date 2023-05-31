# DIP Final Project

if [ ! -d "./result" ]
then
   mkdir ./result 
fi

if [ ! -d "./result/test1" ]
then
   mkdir ./result/test1
fi

# python3 ImageInpainting.py --input ./dataset/data9/image.png --mask ./dataset/data9/mask.png --output ./result/result9.png --patch_size 7
# python3 ImageInpainting.py --input ./dataset/data8/image.png --mask ./dataset/data8/mask.png --output ./result/result8.png --patch_size 9
python3 ImageInpainting.py --input ./dataset/data8/image.png --mask ./dataset/data8/mask.png --output ./result/result8-1.png --patch_size 5

