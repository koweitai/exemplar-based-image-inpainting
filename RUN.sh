# DIP Final Project

if [ ! -d "./result" ]
then
   mkdir ./result 
fi

if [ ! -d "./result/result9" ]
then
   mkdir ./result/result9
fi

# python3 ImageInpainting.py --input ./dataset/data9/image.png --mask ./dataset/data9/mask.png --output ./result/result9.png --patch_size 7
# python3 ImageInpainting.py --input ./dataset/data8/image.png --mask ./dataset/data8/mask.png --output ./result/result8.png --patch_size 9
# python3 ImageInpainting.py --input ./dataset/data10/image.png --mask ./dataset/data10/mask-2.png --output ./result/test5/result10.png --patch_size 5
# python3 ImageInpainting.py --input ./dataset/data4/image.jpg --mask ./dataset/data4/mask.jpg --output ./result/result4_test/result4.png --patch_size 7
python3 ImageInpainting.py --input ./dataset/data11/image.png --mask ./dataset/data11/mask.png --output ./result/result11/result11.png --patch_size 9


