import subprocess
import sys
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse

special='trial_50ep'
DF_filename='DF_'+special+'.pickle'
TS_filename='TS_'+special+'.pickle'


parser=argparse.ArgumentParser(description = 'Visulation and tuning DF vs TS comparison')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--save', default=False, type=bool, help='Whether or not to save all generated figs')
args=parser.parse_args()

print ('Starting DF to be saved as: '+DF_filename)
subprocess.run(f'python3 train.py --method DF --sv {DF_filename} --epochs {args.epochs}', shell=True)
print ('Starting TS to be saved as: '+TS_filename)
subprocess.run(f'python3 train.py --method TS --sv {TS_filename} --epochs {args.epochs}', shell=True)
print ('BOTH DONE')


with open (DF_filename, 'rb') as df_file:
    df_output=pickle.load(df_file)

with open (TS_filename, 'rb') as ts_file:
    ts_output=pickle.load(ts_file)


modes=['train', 'val', 'test']


for mode in modes:
    
    num_epochs= len(df_output[0][mode])-1 ## Last entry is the OPE if GT is perfectly known

    #Loss figure
    plt.figure()
    
    plt.plot(range(num_epochs), df_output[0][mode][:num_epochs], label='DF')
    plt.plot(range(num_epochs), ts_output[0][mode][:num_epochs], label='TS')
    plt.legend()
    plt.title(mode+' Loss comparison')
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_loss.png')
    plt.show()

    #Ope figure
    plt.figure()
    plt.plot(range(num_epochs), df_output[1][mode][:num_epochs], label='DF')
    plt.plot(range(num_epochs), ts_output[1][mode][:num_epochs], label='TS')
    plt.plot(range(num_epochs), [df_output[1][mode][num_epochs] for _ in range(num_epochs)], '--', label='DF_GT_known')
    plt.plot(range(num_epochs), [ts_output[1][mode][num_epochs] for _ in range(num_epochs)], '--', label='TS_GT_known')
    plt.legend()
    plt.title(mode+' OPE comparison')
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_OPE.png')
    plt.show()


