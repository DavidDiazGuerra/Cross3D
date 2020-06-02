"""
	Python script to train the Cross3D model and analyze its performance.

	File name: 1sourceTracking_Cross3D.py
	Author: David Diaz-Guerra
	Date creation: 05/2020
	Python Version: 3.8
	Pytorch Version: 1.4.0
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

import acousticTrackingDataset as at_dataset
import acousticTrackingLearners as at_learners
import acousticTrackingModels as at_models
from acousticTrackingDataset import Parameter


# %% Parameters
res_the = 32  # Maps resolution (elevation)
res_phi = 64  # Maps resolution (azimuth)

K = 4096  # Window size
fs = 16000

N = 12
array_setup = at_dataset.benchmark2_array_setup
array_name = 'robot'  # Only for the output filenames
array_locata_name = 'benchmark2'  # Name of the array in the LOCATA dataset

model_name = 'cross3D'  # Only for the output filenames, change it also in Network declaration cell
extra_notes = ''
suffix = model_name + '_' + array_name + '_K' + str(K) + '_' + str(res_the) + 'x' + str(res_phi)
if extra_notes is not None and extra_notes != '':
	suffix += '_' + extra_notes


# %% Dataset
T = 20 # Trajectory length (s)
path_train = "datasets/LibriSpeech/train-clean-100"
path_test = "datasets/LibriSpeech/test-clean"
corpusDataset_train = at_dataset.LibriSpeechDataset(path_train, T, return_vad=True)
corpusDataset_test = at_dataset.LibriSpeechDataset(path_test, T, return_vad=True)

windowing = at_dataset.Windowing(K, K*3//4, window=np.hanning)

dataset_train = at_dataset.RandomTrajectoryDataset(
	sourceDataset = corpusDataset_train,
	room_sz = Parameter([3,3,2.5], [10,8,6]),  	# Random room sizes from 3x3x2.5 to 10x8x6 meters
	T60 = Parameter(0.2, 1.3),					# Random reverberation times from 0.2 to 1.3 seconds
	abs_weights = Parameter([0.5]*6, [1.0]*6),  # Random absorption weights ratios between walls
	array_setup = array_setup,
	array_pos = Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]), # Ensure a minimum separation between the array and the walls
	SNR = Parameter(30), 	# Start the simulation with a low level of omnidirectional noise
	nb_points = 156,		# Simulate 156 RIRs per trajectory (independent from the SRP-PHAT window length
	transforms = [windowing]
)
dataset_test = at_dataset.RandomTrajectoryDataset(  # The same setup than for training but with other source signals
	sourceDataset = corpusDataset_test,
	room_sz = Parameter([3,3,2.5], [10,8,6]),
	T60 = Parameter(0.2, 1.3),
	abs_weights = Parameter([0.5]*6, [1.0]*6),
	array_setup = array_setup,
	array_pos = Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]),
	SNR = Parameter(30),
	nb_points = 156,
	transforms = [windowing]
)


# %% Network declaration
cr_deep = int(min(4, np.log2(min(res_the, res_phi))))  # For low resolution maps it is not possible to perform 4 cross layers
net = at_models.Cross3D(res_the, res_phi, cr_deep=cr_deep)
learner = at_learners.OneSourceTrackingFromMapsLearner(net, N, K, res_the, res_phi, array_setup.mic_pos, fs,
													   arrayType=array_setup.arrayType, cat_maxCoor=True, apply_vad=True)
learner.cuda()


# %% Network training
trajectories_per_batch = 5
trajectories_per_gpu_call = 5
lr = 0.0001
nb_epoch = 80

print('Training network...')
for epoch_idx in range(nb_epoch):
	print('\nEpoch {}/{}:'.format(epoch_idx+1, nb_epoch))
	sys.stdout.flush()

	learner.train_epoch(dataset_train, trajectories_per_batch, trajectories_per_gpu_call, lr=lr, epoch=epoch_idx)
	loss_test, rmsae_test = learner.test_epoch(dataset_test, trajectories_per_gpu_call)
	print('Test loss: {:.4f},   Test rmsae: {:.2f}deg'.format(loss_test, rmsae_test) )
	sys.stdout.flush()

	if epoch_idx == 19:
		print('\nDecreasing SNR')
		dataset_train.SNR = Parameter(5, 30)  	# Random SNR between 5dB and 30dB after the model has started to converge
		dataset_test.SNR = Parameter(5, 30)  	# Random SNR between 5dB and 30dB after the model has started to converge
		trajectories_per_batch = 10  			# Increase the batch size
		lr = lr/10								# Decrease the learning rate

print('\nTraining finished\n')


# %% Save model
print('Saving model...')
torch.save(net.state_dict(), 'models/' + '1sourceTracking_' + suffix + '_model.bin')
print('Model saved.\n')
sys.stdout.flush()


# %% Load model
cr_deep = int(min(4, np.log2(min(res_the, res_phi))))
net = at_models.Cross3D(res_the, res_phi, cr_deep=cr_deep)
net.load_state_dict(torch.load('models/' + '1sourceTracking_' + suffix + '_model.bin'))
learner = at_learners.OneSourceTrackingFromMapsLearner(net, N, K, res_the, res_phi, array_setup.mic_pos, fs,
													   arrayType=array_setup.arrayType, cat_maxCoor=True, apply_vad=True)
learner.cuda()


# %% Analyze results
print("Analyzing results for several reverberation times")
sys.stdout.flush()
T60 = np.array((0, 0.3, 0.6, 0.9, 1.2, 1.5))  	# Reverberation times to analyze
SNR = np.array((5, 15, 30))						# SNRs to analyze
acoustic_scenes = np.empty((len(T60), len(SNR)), dtype=object)  # To store the analyzed acoustic scenes
rmsae = np.zeros((len(T60), len(SNR)))							# Root Mean Saqure Angular Error (degrees) of the model
rmsae_srpMax = np.zeros((len(T60), len(SNR)))					# RMSAE of the SRP-PHAT maximums
rmsae_noSilences = np.zeros((len(T60), len(SNR)))				# RMSAE of the model without include silent frames
rmsae_srpMax_noSilences = np.zeros((len(T60), len(SNR)))		# RMSAE of the SRP-PHAT maximums without include silent frames
trajectories_per_batch = 5

for i in range(len(T60)):
	for j in range(len(SNR)):
		print('Analyzing scenes with T60=' + str(T60[i]) + 's and SNR=' + str(SNR[j]) + 'dB')
		sys.stdout.flush()
		dataset_test.T60 = Parameter(T60[i])
		dataset_test.SNR = Parameter(SNR[j])
		acoustic_scenes[i,j], rmsae[i,j] = learner.predict_dataset(dataset_test, trajectories_per_batch, return_rmsae=True, save_x=True, x_filed_name='maps')
		# In order to get more accurate RMSAEs you can use dataset_train, which simulates a higher number of trajectories,
		# but it takes more time and has a bigger memory consumption

		for k in range(len(acoustic_scenes[i,j])):
			rmsae_srpMax[i,j] += acoustic_scenes[i,j][k].findMapMaximum() / len(acoustic_scenes[i,j])
			rmsae_srpMax_noSilences[i,j] += acoustic_scenes[i,j][k].findMapMaximum(exclude_silences=True)/len(acoustic_scenes[i,j])
			rmsae_noSilences[i,j] += acoustic_scenes[i,j][k].get_rmsae(exclude_silences=True)/len(acoustic_scenes[i,j])

		acoustic_scenes[i,j] = acoustic_scenes[i,j][:10] # Store only 10 scenes, they include the source signals
														 # and the maps, so they weights quite a lot


# %% Save analyzed results
print("Saving results")
sys.stdout.flush()
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_T60.npy', T60)
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_SNR.npy', SNR)
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae.npy', rmsae)
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae_srpMax.npy', rmsae_srpMax)
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae_noSilences.npy', rmsae_noSilences)
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae_srpMax_noSilences.npy', rmsae_srpMax_noSilences)
np.save('results/' + '1sourceTracking_' + suffix + '_predictions_acoustic_scenes.npy', acoustic_scenes)
print("Results saved\n")
sys.stdout.flush()


# %% Load analyzed results
T60 = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_T60.npy')
SNR = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_SNR.npy')
rmsae = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae.npy')
rmsae_srpMax = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae_srpMax.npy')
rmsae_noSilences = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae_noSilences.npy')
rmsae_srpMax_noSilences = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_rmsae_srpMax_noSilences.npy')
acoustic_scenes = np.load('results/' + '1sourceTracking_' + suffix + '_predictions_acoustic_scenes.npy', allow_pickle=True)


# %% Plot analyzed results
legend = []
style = ['-', '--', '-.', ':']
for j in range(len(SNR)):
	plt.plot(T60, np.stack((rmsae_noSilences[:, -j-1], rmsae_srpMax_noSilences[:, -j-1]), axis=-1), style[j])
	legend += ['3D CNN (SNR={})'.format(SNR[-j-1]), 'SRP-PHAT max (SNR={})'.format(SNR[-j-1])]
	plt.gca().set_prop_cycle(None)
plt.legend(legend)
plt.xlabel('T60 [s]')
plt.ylabel('rmsae [º]')
plt.title(suffix)
plt.show()

legend = []
style = ['-', '--', '-.', ':']
for j in range(len(SNR)):
	plt.plot(T60, np.stack((rmsae[:, -j-1]), axis=-1), style[j])
	legend += ['Cross3D (SNR={})'.format(SNR[-j-1])]
	plt.gca().set_prop_cycle(None)
plt.legend(legend)
plt.xlabel('T60 [s]')
plt.ylabel('rmsae [º]')
plt.title(suffix)
plt.show()


# %% Analyze LOCATA dataset
if array_locata_name != '' and array_locata_name is not None:
	print("Analyzing LOCATA dataset")

	path_locata = "datasets/LibriSpeech/dev/"
	windowing = at_dataset.Windowing(K, K * 3 // 4, window=np.hanning)
	dataset_locata = at_dataset.LocataDataset(path_locata, array_locata_name, fs, dev=True,
											  tasks=(1,3,5), transforms=[windowing])
	acoustic_scenes_locata = learner.predict_dataset(dataset_locata, 1, save_x=True, x_filed_name='maps')

	rmsae = np.zeros(len(dataset_locata))			# RMSAE of the model for each LOCATA recording including silent frames
	rmsae_srpMax = np.zeros(len(dataset_locata))  	# RMSAE of the SRP-PHAT maximums for each LOCATA recording without silent frames
	for i in range(len(dataset_locata)):
		rmsae_srpMax[i] = acoustic_scenes_locata[i].findMapMaximum(exclude_silences=True)
		rmsae[i] = acoustic_scenes_locata[i].get_rmsae(exclude_silences=False)

	print('3D CNN rmsae (with silences): ' + str(rmsae))
	print('SRP-PHAT maximum rmsae (without silences): ' + str(rmsae_srpMax) + '\n')

