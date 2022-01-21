import subprocess
import sys
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse

special='placeholder'


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

    random_metrics = [[ts_outputs[sd-args.seed][i][mode][0] for i in range(3)] for sd in range(args.seed, args.seed+args.tr)]
    random_mean, random_ste = np.mean(random_metrics, axis=0), np.std(random_metrics, axis=0) / np.sqrt(len(ts_outputs))
    
    ### Loss figure
    plt.figure()
    
    lw = 3

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

    plt.plot(range(num_epochs), df_is_means, label='DF-IS', color='#4285F4', lw=lw)
    plt.fill_between(range(num_epochs), df_is_means-df_is_errors, df_is_means+df_is_errors, alpha=0.2, color='#4285F4')
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS', color='#F4B400', lw=lw)
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2, color='#F4B400')

    plt.hlines(random_mean[0], xmin=0, xmax=num_epochs, colors='#DB4437', lw=lw, linestyle='dashed', label='random')

    plt.legend(fontsize=18)
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

    plt.plot(range(num_epochs), df_is_means, label='DF-IS', color='#4285F4', lw=lw)
    plt.fill_between(range(num_epochs), df_is_means-df_is_errors, df_is_means+df_is_errors, alpha=0.2, color='#4285F4')
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS', color='#F4B400', lw=lw)
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2, color='#F4B400')

    plt.hlines(random_mean[1], xmin=0, xmax=num_epochs, colors='#DB4437', lw=lw, linestyle='dashed', label='random')
    
    plt.legend(fontsize=18)
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

    plt.plot(range(num_epochs), df_is_means, label='DF-IS', color='#4285F4', lw=lw)
    plt.fill_between(range(num_epochs), df_is_means-df_is_errors, df_is_means+df_is_errors, alpha=0.2, color='#4285F4')
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS', color='#F4B400', lw=lw)
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2, color='#F4B400')

    plt.hlines(random_mean[1], xmin=0, xmax=num_epochs, colors='#DB4437', lw=lw, linestyle='dashed', label='random')
    
    plt.legend(fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('OPE-Sim', fontsize=18)
    plt.title(mode+' Sim-OPE comparison', fontsize=18)
    if args.save:
        plt.savefig('./figs/'+special+'_'+mode+'_OPE_SIM.png')
    plt.show()


  ### Table information
  ts_selected_metrics = []
  df_is_selected_metrics = []
  df_sim_selected_metrics = []
  for sd in range(args.seed, args.seed+args.tr):
    # Two-stage selected epoch
    print (sd, len(ts_outputs))
    ts_selected_epoch = -2 # np.argmin(ts_outputs[sd-args.seed][0]['val'][:-1]) # loss metric
    ts_selected_metrics.append([ts_outputs[sd-args.seed][i]['test'][ts_selected_epoch] for i in range(3)])

    # DF-IS selected epoch
    df_is_selected_epoch = -2 # np.argmax(df_is_outputs[sd-args.seed][1]['val'][:-1]) # maximize IS OPE  metric
    df_is_selected_metrics.append([df_is_outputs[sd-args.seed][i]['test'][df_is_selected_epoch] for i in range(3)])

    # DF-sim selected epoch
    df_sim_selected_epoch = np.argmax(df_sim_outputs[sd-args.seed][2]['val'][:-1]) # Maximize SIM OPE
    df_sim_selected_metrics.append([df_sim_outputs[sd-args.seed][i]['test'][df_sim_selected_epoch] for i in range(3)])
  
  ts_selected_metrics=np.array(ts_selected_metrics)
  df_is_selected_metrics=np.array(df_is_selected_metrics)
  df_sim_selected_metrics=np.array(df_sim_selected_metrics)

  ts_test_mean, ts_test_ste         = np.mean(ts_selected_metrics, axis=0), np.std(ts_selected_metrics, axis=0) / np.sqrt(len(ts_outputs))
  df_is_test_mean, df_is_test_ste   = np.mean(df_is_selected_metrics, axis=0), np.std(df_is_selected_metrics, axis=0) / np.sqrt(len(df_is_outputs))
  df_sim_test_mean, df_sim_test_ste = np.mean(df_sim_selected_metrics, axis=0), np.std(df_sim_selected_metrics, axis=0) / np.sqrt(len(df_sim_outputs))

  print('Random test metrics mean (Loss/IS OPE/Sim OPE): ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$'.format(random_mean[0], random_ste[0], random_mean[1], random_ste[1], random_mean[2], random_ste[2])) # only valid after 0119
  print('Two-stage test metrics mean (Loss/IS OPE/Sim OPE): ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$'.format(ts_test_mean[0], ts_test_ste[0], ts_test_mean[1], ts_test_ste[1], ts_test_mean[2], ts_test_ste[2]))
  print('DF-IS test metrics mean (Loss/IS OPE/ Sim OPE): ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$'.format(df_is_test_mean[0], df_is_test_ste[0], df_is_test_mean[1], df_is_test_ste[1], df_is_test_mean[2], df_is_test_ste[2]))
  print('DF-sim test metrics mean (Loss/IS OPE/Sim OPE): ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$, ${:.1f}\pm{:.1f}$'.format(df_sim_test_mean[0], df_sim_test_ste[0], df_sim_test_mean[1], df_sim_test_ste[1], df_sim_test_mean[2], df_sim_test_ste[2]))
