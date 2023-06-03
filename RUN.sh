# DIP Final Project

if [ ! -d "./result" ]
then
   mkdir ./result 
fi

if [ ! -d "./result/test2" ]
then
   mkdir ./result/test2
fi

# python3 ImageInpainting.py --input ./dataset/data9/image.png --mask ./dataset/data9/mask.png --output ./result/result9.png --patch_size 7
# python3 ImageInpainting.py --input ./dataset/data8/image.png --mask ./dataset/data8/mask.png --output ./result/result8.png --patch_size 9
python3 ImageInpainting.py --input ./dataset/data10/image.png --mask ./dataset/data10/mask-2.png --output ./result/result10.png --patch_size 5

