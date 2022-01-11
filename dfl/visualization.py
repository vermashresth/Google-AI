import subprocess
import sys
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse

special='0110'


parser=argparse.ArgumentParser(description = 'Visulation and tuning DF vs TS comparison')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--instances', default=10, type=int, help='Number of instances')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--save', default=0, type=int, help='Whether or not to save all generated figs. Put 0 for False, 1 for True')
parser.add_argument('--plot', default=0, type=int, help='Whether or not to create plots. Put 0 for False, 1 for True')
parser.add_argument('--compute', default=0, type=int, help='Whether or not to run new experiments. Put 0 for False, 1 for True')
parser.add_argument('--tr', default=1, type=int, help='Number of trials to be run starting with seed value entered for seed.')
parser.add_argument('--name', default='.', type=str, help='Special string name.')
args=parser.parse_args()


args.save=bool(args.save)
args.plot=bool(args.plot)
args.compute=bool(args.compute)

if not args.name=='.':
  special=args.name
  print ("Using special save string: ", special)

if args.compute:
  ### Launch new computational experiments for the specified settings if True 
  for sd in range(args.seed, args.seed+args.tr):
 
    DF_filename='./results/DF_'+special+'_sd_'+str(sd)+'.pickle'
    TS_filename='./results/TS_'+special+'_sd_'+str(sd)+'.pickle'

    print ('Starting seed: ', sd)
    print ('Starting DF to be saved as: '+DF_filename)
    subprocess.run(f'python3 train.py --method DF --sv {DF_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd}', shell=True)
    print ('Starting TS to be saved as: '+TS_filename)
    subprocess.run(f'python3 train.py --method TS --sv {TS_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd}', shell=True)
    print ('BOTH DONE')



if args.plot:
  ### Plot figures for the specified settings if True 
  modes=['train', 'val', 'test']
  for mode in modes:
  
    df_outputs=[]
    ts_outputs=[]
    for sd in range(args.seed, args.seed+args.tr):

      DF_filename='./results/DF_'+special+'_sd_'+str(sd)+'.pickle'
      TS_filename='./results/TS_'+special+'_sd_'+str(sd)+'.pickle'

      with open (DF_filename, 'rb') as df_file:
          df_outputs.append(pickle.load(df_file))

      with open (TS_filename, 'rb') as ts_file:
          ts_outputs.append(pickle.load(ts_file))


    num_epochs= len(df_outputs[0][0][mode])-1 ## Last entry is the OPE if GT is perfectly known

    ### Loss figure
    plt.figure()
    
    df_means=[]
    df_errors=[]
    ts_means=[]
    ts_errors=[]
    
    for epoch in range(num_epochs):
        df_outputs_for_this_epoch=np.array([item[0][mode][epoch] for item in df_outputs])
        ts_outputs_for_this_epoch=np.array([item[0][mode][epoch] for item in ts_outputs])

        df_means.append(np.mean(df_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))

        df_errors.append(np.std(df_outputs_for_this_epoch)/np.sqrt(len(df_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    df_means=np.array(df_means)
    df_errors=np.array(df_errors)
    ts_means=np.array(ts_means)
    ts_errors=np.array(ts_errors)

    plt.plot(range(num_epochs), df_means, label='DF')
    plt.fill_between(range(num_epochs), df_means-df_errors, df_means+df_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS')
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Intermediate Loss')
    plt.title(mode+' Loss comparison')
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_loss.png')
    plt.show()

    ### IS OPE figure
    plt.figure()
    
    df_means=[]
    df_errors=[]
    ts_means=[]
    ts_errors=[]
    
    for epoch in range(num_epochs+1):
        df_outputs_for_this_epoch=np.array([item[1][mode][epoch] for item in df_outputs])
        ts_outputs_for_this_epoch=np.array([item[1][mode][epoch] for item in ts_outputs])

        df_means.append(np.mean(df_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))

        df_errors.append(np.std(df_outputs_for_this_epoch)/np.sqrt(len(df_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    df_means=np.array(df_means)
    df_errors=np.array(df_errors)
    ts_means=np.array(ts_means)
    ts_errors=np.array(ts_errors)

    plt.plot(range(num_epochs), df_means[:num_epochs], label='DF')
    plt.fill_between(range(num_epochs), df_means[:num_epochs]-df_errors[:num_epochs], df_means[:num_epochs]+df_errors[:num_epochs], alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means[:num_epochs], label='TS')
    plt.fill_between(range(num_epochs), ts_means[:num_epochs]-ts_errors[:num_epochs], ts_means[:num_epochs]+ts_errors[:num_epochs], alpha=0.2)
    
    plt.errorbar(range(num_epochs), [df_means[num_epochs] for _ in range(num_epochs)], yerr=[df_errors[num_epochs] for _ in range(num_epochs)],  fmt='--', capsize=4, label='DF_GT_known')
    plt.errorbar(range(num_epochs), [ts_means[num_epochs] for _ in range(num_epochs)], yerr=[ts_errors[num_epochs] for _ in range(num_epochs)] , fmt='--', capsize=4, label='TS_GT_known')
 
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('IS OPE')
    plt.title(mode+' OPE (importance sampling) comparison')
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_OPE_IS.png')
    plt.show()


    ### SIM  OPE figure
    plt.figure()
    
    df_means=[]
    df_errors=[]
    ts_means=[]
    ts_errors=[]
    
    for epoch in range(num_epochs+1):
        df_outputs_for_this_epoch=np.array([item[2][mode][epoch] for item in df_outputs])
        ts_outputs_for_this_epoch=np.array([item[2][mode][epoch] for item in ts_outputs])

        df_means.append(np.mean(df_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))

        df_errors.append(np.std(df_outputs_for_this_epoch)/np.sqrt(len(df_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    df_means=np.array(df_means)
    df_errors=np.array(df_errors)
    ts_means=np.array(ts_means)
    ts_errors=np.array(ts_errors)

    plt.plot(range(num_epochs), df_means[:num_epochs], label='DF')
    plt.fill_between(range(num_epochs), df_means[:num_epochs]-df_errors[:num_epochs], df_means[:num_epochs]+df_errors[:num_epochs], alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means[:num_epochs], label='TS')
    plt.fill_between(range(num_epochs), ts_means[:num_epochs]-ts_errors[:num_epochs], ts_means[:num_epochs]+ts_errors[:num_epochs], alpha=0.2)
    
    plt.errorbar(range(num_epochs), [df_means[num_epochs] for _ in range(num_epochs)], yerr=[df_errors[num_epochs] for _ in range(num_epochs)],  fmt='--', capsize=4, label='DF_GT_known')
    plt.errorbar(range(num_epochs), [ts_means[num_epochs] for _ in range(num_epochs)], yerr=[ts_errors[num_epochs] for _ in range(num_epochs)] , fmt='--', capsize=4, label='TS_GT_known')
 
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('SIM OPE')
    plt.title(mode+' OPE (SIM) comparison')
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_OPE_SIM.png')
    plt.show()

