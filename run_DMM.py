import argparse
import os
from DMM_config import d

parser = argparse.ArgumentParser()

parser.add_argument('-size', type=int, default=1)
parser.add_argument('-data', default=['Exchange'], required=True, type=str, nargs='+')
parser.add_argument('-model', default=['DMM'], type=str, nargs='+')
parser.add_argument('-device', default=0, type=int)
parser.add_argument('-mask_type', default=['MAR'], required=True, type=str, nargs='+')
parser.add_argument('-DMM_type', default=['MAR'], required=True, type=str, nargs='+')
parser.add_argument('-mask_rate', default=[0.2], required=True, type=float, nargs='+')
parser.add_argument('-seed', default=[2024], type=int, nargs='+')
parser.add_argument('-train_mode', default=0, type=int)

args = parser.parse_args()

comand_list = []

for mask_type in args.mask_type:
    for seed in args.seed:
        for data in args.data:
            for model in args.model:
                for DMM_type in args.DMM_type:
                    for mask_rate in args.mask_rate:
                        command = f"{d[data][model][mask_type][mask_rate]} --train_mode {args.train_mode} --mask_type {mask_type} --DMM_type {DMM_type} --mask_rate {mask_rate} --seed {seed} --gpu {args.device}"
                        comand_list.append(command)

i = 0
while i + args.size <= len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(args.size)).rstrip().rstrip('&')
    os.system(new_comand)
    i = i + args.size

if i < len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(len(comand_list) - i)).rstrip().rstrip('&')
    os.system(new_comand)
