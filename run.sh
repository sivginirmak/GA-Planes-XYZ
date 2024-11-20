#!/bin/bash

exp_name=gaplane-semiconvex
exp_type=eval_iou #train eval_iou
gpu=0
supervision=proj # sc (3D) or proj (2D)
model_type=GAplanes #GAplanes triplane
pe=0
convex=0
semi_convex=1
save_on=1
n_epochs=4000
display_rate=250

#gaplane small
Cl=36
Cp=24
Cv=8
Nl=128
Np=32
Nv=24
#triplane small
# Cl=24
# Cp=4
# Cv=8
# Nl=128
# Np=128
# Nv=32

comments=first-experiment
seed=0

python run_exp.py --exp_name $exp_name \
                  --supervision $supervision \
                  --model_type $model_type \
                  --gpu $gpu \
                    --pe $pe \
                    --convex $convex \
                    --semi_convex $semi_convex \
                    --save_on $save_on \
                    --n_epochs $n_epochs \
                    --display_rate $display_rate \
                    --Cp $Cp \
                    --Cl $Cl \
                    --Cv $Cv \
                    --Nv $Nv \
                    --Np $Np \
                    --Nl $Nl \
                    --exp_type $exp_type \
                    --seed $seed \
                    --comments $comments 



