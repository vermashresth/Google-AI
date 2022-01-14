import subprocess
import sys
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse

special='0113'


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
 
    DF_IS_filename='./results/DF_IS_'+special+'_sd_'+str(sd)+'.pickle'
    DF_SIM_filename='./results/DF_SIM_'+special+'_sd_'+str(sd)+'.pickle'
    TS_filename='./results/TS_'+special+'_sd_'+str(sd)+'.pickle'

    print ('Starting seed: ', sd)
    print ('Starting DF Importance Sampling to be saved as: '+DF_IS_filename)
    subprocess.run(f'python3 train.py --method DF --sv {DF_IS_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --ope {"IS"}', shell=True)
    print ('Starting DF Simu based to be saved as: '+DF_SIM_filename)
    subprocess.run(f'python3 train.py --method DF --sv {DF_SIM_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --ope {"sim"}', shell=True)
    print ('Starting TS to be saved as: '+TS_filename)
    subprocess.run(f'python3 train.py --method TS --sv {TS_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd}', shell=True)
    print ('ALL THREE DONE')



if args.plot:
  ### Plot figures for the specified settings if True 
  modes=['train', 'val', 'test']
  for mode in modes:
  
    df_is_outputs=[]
    df_sim_outputs=[]
    ts_outputs=[]
    for sd in range(args.seed, args.seed+args.tr):

      DF_IS_filename='./results/DF_IS_'+special+'_sd_'+str(sd)+'.pickle'
      DF_SIM_filename='./results/DF_SIM_'+special+'_sd_'+str(sd)+'.pickle'
      TS_filename='./results/TS_'+special+'_sd_'+str(sd)+'.pickle'

      with open (DF_IS_filename, 'rb') as df_is_file:
          df_is_outputs.append(pickle.load(df_is_file))

      with open (DF_SIM_filename, 'rb') as df_sim_file:
          df_sim_outputs.append(pickle.load(df_sim_file))

      with open (TS_filename, 'rb') as ts_file:
          ts_outputs.append(pickle.load(ts_file))


    num_epochs= len(df_is_outputs[0][0][mode])-1 ## Last entry is the OPE if GT is perfectly known

    ### Loss figure
    plt.figure()
    
    df_is_means=[]
    df_is_errors=[]
    df_sim_means=[]
    df_sim_errors=[]
    ts_means=[]
    ts_errors=[]
    
    for epoch in range(num_epochs):
        df_is_outputs_for_this_epoch=np.array([item[0][mode][epoch] for item in df_is_outputs])
        df_sim_outputs_for_this_epoch=np.array([item[0][mode][epoch] for item in df_sim_outputs])
        ts_outputs_for_this_epoch=np.array([item[0][mode][epoch] for item in ts_outputs])

        df_is_means.append(np.mean(df_is_outputs_for_this_epoch))
        df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))

        df_is_errors.append(np.std(df_is_outputs_for_this_epoch)/np.sqrt(len(df_is_outputs_for_this_epoch)))
        df_sim_errors.append(np.std(df_sim_outputs_for_this_epoch)/np.sqrt(len(df_sim_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    df_is_means=np.array(df_is_means)
    df_is_errors=np.array(df_is_errors)
    df_sim_means=np.array(df_sim_means)
    df_sim_errors=np.array(df_sim_errors)
    
    ts_means=np.array(ts_means)
    ts_errors=np.array(ts_errors)

    plt.plot(range(num_epochs), df_is_means, label='DF-IS')
    plt.fill_between(range(num_epochs), df_is_means-df_is_errors, df_is_means+df_is_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS')
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2)

    plt.legend()
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Intermediate Loss', fontsize=18)
    plt.title(mode+' Loss comparison', fontsize=18)
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_loss.png')
    plt.show()

    ### IS OPE figure
    plt.figure()
    
    df_is_means=[]
    df_is_errors=[]
    df_sim_means=[]
    df_sim_errors=[]
    ts_means=[]
    ts_errors=[]
    
    for epoch in range(num_epochs):
        df_is_outputs_for_this_epoch=np.array([item[1][mode][epoch] for item in df_is_outputs])
        df_sim_outputs_for_this_epoch=np.array([item[1][mode][epoch] for item in df_sim_outputs])
        ts_outputs_for_this_epoch=np.array([item[1][mode][epoch] for item in ts_outputs])

        df_is_means.append(np.mean(df_is_outputs_for_this_epoch))
        df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))

        df_is_errors.append(np.std(df_is_outputs_for_this_epoch)/np.sqrt(len(df_is_outputs_for_this_epoch)))
        df_sim_errors.append(np.std(df_sim_outputs_for_this_epoch)/np.sqrt(len(df_sim_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    df_is_means=np.array(df_is_means)
    df_is_errors=np.array(df_is_errors)
    df_sim_means=np.array(df_sim_means)
    df_sim_errors=np.array(df_sim_errors)
    
    ts_means=np.array(ts_means)
    ts_errors=np.array(ts_errors)

    plt.plot(range(num_epochs), df_is_means, label='DF-IS')
    plt.fill_between(range(num_epochs), df_is_means-df_is_errors, df_is_means+df_is_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS')
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2)

    plt.legend()
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('IS-OPE', fontsize=18)
    plt.title(mode+' IS-OPE comparison', fontsize=18)
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_OPE_IS.png')
    plt.show()

    ### SIM-OPE figure
    plt.figure()
    
    df_is_means=[]
    df_is_errors=[]
    df_sim_means=[]
    df_sim_errors=[]
    ts_means=[]
    ts_errors=[]
    
    for epoch in range(num_epochs):
        df_is_outputs_for_this_epoch=np.array([item[2][mode][epoch] for item in df_is_outputs])
        df_sim_outputs_for_this_epoch=np.array([item[2][mode][epoch] for item in df_sim_outputs])
        ts_outputs_for_this_epoch=np.array([item[2][mode][epoch] for item in ts_outputs])

        df_is_means.append(np.mean(df_is_outputs_for_this_epoch))
        df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))

        df_is_errors.append(np.std(df_is_outputs_for_this_epoch)/np.sqrt(len(df_is_outputs_for_this_epoch)))
        df_sim_errors.append(np.std(df_sim_outputs_for_this_epoch)/np.sqrt(len(df_sim_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    df_is_means=np.array(df_is_means)
    df_is_errors=np.array(df_is_errors)
    df_sim_means=np.array(df_sim_means)
    df_sim_errors=np.array(df_sim_errors)
    
    ts_means=np.array(ts_means)
    ts_errors=np.array(ts_errors)

    plt.plot(range(num_epochs), df_is_means, label='DF-IS')
    plt.fill_between(range(num_epochs), df_is_means-df_is_errors, df_is_means+df_is_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS')
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2)

    plt.legend()
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('OPE-Sim', fontsize=18)
    plt.title(mode+' Sim-OPE comparison', fontsize=18)
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_OPE_SIM.png')
    plt.show()


