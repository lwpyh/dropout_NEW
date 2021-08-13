#!/bin/bash
set -x
params_alpha=( 888 777 666 555 444 111 999)
#params_lamda=( 0\.8 1\.0 1\.2)

#for j in  ${params_lamda[@]}
#do

for i in ${params_alpha[@]}
do
        echo "alpha is $i"

        sed -i "325s/config\[\"seed\"\]\=*[0-9]*\.*[0-9]*/config\[\"seed\"\]\=$j/" ./train_image_sharpen.py
        #sed -i "29s/lamda\=*[0-9]*\.*[0-9]*/lamda\=$j/" ./DANN_Alexnet.py
        #sed -i "325s/config\[\"tradeoff\"\]\=*[0-9]*\.*[0-9]*/config\[\"tradeoff\"\]\=$i/" ./train_pada1.py
        CUDA_VISIBLE_DEVICES=0 python ./train_mcdrop-all.py --seed $i

#done
done

        
    

