"""
	Learner classes to train the models and perform inferences.

	File name: acousticTrackingLearners.py
	Author: David Diaz-Guerra
	Date creation: 05/2020
	Python Version: 3.8
	Pytorch Version: 1.4.0
"""

import numpy as np
import torch
import torch.optim as optim
import webrtcvad
from abc import ABC, abstractmethod
from tqdm import trange

from utils import sph2cart, cart2sph, rms_angular_error_deg
import acousticTrackingModules as at_modules


class OneSourceTrackingLearner(ABC):
	""" Abstract class to the routines to train the one source tracking models and perform inferences.
	"""
	def __init__(self, model):
		self.model = model
		self.cuda_activated = False
		super().__init__()

	def cuda(self):
		""" Move the model to the GPU and perform the training and inference there.
		"""
		self.model.cuda()
		self.cuda_activated = True

	def cpu(self):
		""" Move the model back to the CPU and perform the training and inference here.
		"""
		self.model.cpu()
		self.cuda_activated = False

	@abstractmethod
	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" To be implemented in each learner according to input of their models
		"""
		pass

	def train_epoch(self, dataset, trajectories_per_batch, trajectories_per_gpu_call=5, lr=0.0001, epoch=None):
		""" Train the model with an epoch of the dataset.
		"""
		assert trajectories_per_batch % trajectories_per_gpu_call == 0

		avg_loss = 0
		avg_beta = 0.99

		self.model.train()  # set the model in "training mode"
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		n_trajectories = len(dataset)

		trajectory_idx = 0
		optimizer.zero_grad()
		pbar = trange(n_trajectories // trajectories_per_gpu_call, ascii=True)
		for gpu_call_idx in pbar:
			if epoch is not None: pbar.set_description('Epoch {}'.format(epoch + 1))

			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(gpu_call_idx * trajectories_per_gpu_call,
																	(gpu_call_idx + 1) * trajectories_per_gpu_call)
			x_batch, DOA_batch = self.data_transformation(mic_sig_batch, acoustic_scene_batch)
			x_batch.requires_grad_()
			DOA_batch_pred_cart = self.model(x_batch).contiguous()
			
			DOA_batch = DOA_batch.contiguous()
			DOA_batch_cart = sph2cart(DOA_batch)
			loss = torch.nn.functional.mse_loss(DOA_batch_pred_cart.view(-1, 3), DOA_batch_cart.view(-1, 3))
			loss.backward()

			trajectory_idx += trajectories_per_gpu_call

			if trajectory_idx % trajectories_per_batch == 0:
				optimizer.step()
				optimizer.zero_grad()

			avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss.item()
			pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (gpu_call_idx + 1)))

			pbar.update()

		optimizer.step()
		optimizer.zero_grad()

		del DOA_batch_pred_cart, loss

	def test_epoch(self, dataset, trajectories_per_batch, nb_batchs=None):
		""" Test the model with an epoch of the dataset.
		"""
		self.model.eval()  # set the model in "testing mode"
		with torch.no_grad():
			loss_data = 0
			rmsae_data = 0

			n_trajectories = len(dataset)
			if nb_batchs is None:
				nb_batchs = n_trajectories // trajectories_per_batch

			for idx in range(nb_batchs):
				mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																		(idx + 1) * trajectories_per_batch)
				x_batch, DOA_batch = self.data_transformation(mic_sig_batch, acoustic_scene_batch)
				DOA_batch_pred_cart = self.model(x_batch).contiguous()
				
				DOA_batch = DOA_batch.contiguous()
				DOA_batch_cart = sph2cart(DOA_batch)
				loss_data += torch.nn.functional.mse_loss(DOA_batch_pred_cart.view(-1, 3), DOA_batch_cart.view(-1, 3))
				
				DOA_batch_pred = cart2sph(DOA_batch_pred_cart)
				rmsae_data += rms_angular_error_deg(DOA_batch.view(-1, 2), DOA_batch_pred.view(-1, 2))

			loss_data /= nb_batchs
			rmsae_data /= nb_batchs

			return loss_data, rmsae_data

	def predict_batch(self, mic_sig_batch, vad_batch=None, return_x=False):
		""" Perform the model inference for an input batch.
		You can use return_x=True in order to return the input of the model in addition to the DOA estimation
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = mic_sig_batch.shape[0]
		trajectory_len = mic_sig_batch.shape[1]

		x_batch = self.data_transformation(mic_sig_batch, vad_batch=vad_batch)

		DOA_batch_pred = self.model(x_batch).cpu().detach()

		DOA_batch_pred_cart = DOA_batch_pred.reshape((n_trajectories, trajectory_len, 3)) # For split trajectories
		DOA_batch_pred = cart2sph(DOA_batch_pred_cart)

		if return_x:
			return DOA_batch_pred, x_batch.cpu().detach()
		else:
			return DOA_batch_pred

	def predict_dataset(self, dataset, trajectories_per_batch, nb_batchs=None, return_rmsae=False, save_x=False,
						x_filed_name='netInput'):
		""" Perform the model inference over the whole dataset.
		It outputs the analyzed AcousticScenes with the DOA estimation in the field DOAw_pred.
		It can also include a filed with input of the network (save_x) and return the RMSAE (return_rmsae).
		"""
		self.model.eval()  # set the model in "testing mode"

		n_trajectories = len(dataset)
		if nb_batchs is None:
			nb_batchs = n_trajectories // trajectories_per_batch

		acoustic_scenes = []
		rmsae = 0

		for idx in range(nb_batchs):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx * trajectories_per_batch,
																	(idx + 1) * trajectories_per_batch)

			vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
			if save_x:
				DOA_batch_pred, x_batch = self.predict_batch(mic_sig_batch, vad_batch=vad_batch, return_x=True)
			else:
				DOA_batch_pred = self.predict_batch(mic_sig_batch, vad_batch=vad_batch, return_x=False)

			for i in range(len(acoustic_scene_batch)):
				acoustic_scene_batch[i].DOAw_pred = DOA_batch_pred[i].numpy()
				if save_x:
					if x_filed_name == 'maps':  # Save only the map, not the other channels
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, 0, ...].numpy())
					else:
						setattr(acoustic_scene_batch[i], x_filed_name, x_batch[i, ...].numpy())
				acoustic_scenes.append(acoustic_scene_batch[i])

			if return_rmsae:
				DOA_batch = self.data_transformation(acoustic_scene_batch=acoustic_scene_batch).cpu().contiguous()
				DOA_batch_pred = DOA_batch_pred
				rmsae += rms_angular_error_deg(DOA_batch.view(-1, 2), DOA_batch_pred.view(-1, 2)).item()

		if return_rmsae:
			return acoustic_scenes, rmsae / nb_batchs
		else:
			return acoustic_scenes

	def getNetworkInput_batch(self, mic_sig_batch):
		""" Get the network input for a data batch
		"""
		return self.data_transformation(mic_sig_batch).cpu().detach().numpy()

	def getNetworkInput_dataset(self, dataset, trajectories_per_batch):
		""" Get the network input for a datataset
		"""
		for batch_idx in range(len(dataset) // trajectories_per_batch):
			mic_sig_batch, acoustic_scene_batch = dataset.get_batch(batch_idx * trajectories_per_batch,
																	(batch_idx + 1) * trajectories_per_batch)

			inputs_batch = self.getNetworkInput_batch(mic_sig_batch)
			if batch_idx == 0:
				inputs = np.empty((len(dataset), inputs_batch.shape[1], inputs_batch.shape[2], inputs_batch.shape[3]))
			inputs[batch_idx * trajectories_per_batch:(batch_idx + 1) * trajectories_per_batch, :, :, :] = inputs_batch

		return inputs


class OneSourceTrackingFromMapsLearner(OneSourceTrackingLearner):
	""" Learner for models which use SRP-PHAT maps as input
	"""
	def __init__(self, model, N, K, res_the, res_phi, rn, fs, c=343.0, arrayType='planar', cat_maxCoor=False, apply_vad=False):
		"""
		model: Model to work with
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		res_the: Resolution of the maps in the elevation axis
		res_phi: Resolution of the maps in the azimuth axis
		rn: Position of each microphone relative to te center of the array
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		arrayType: 'planar' or '3D' whether all the microphones are in the same plane (and the maximum DOA elevation is pi/2) or not [default: 'planar']
		cat_maxCoor: Include to the network input tow addition channels with the normalized coordinates of each map maximum [default: False]
		apply_vad: Turn to zero all the map pixels in frames without speech signal [default: False]
		"""
		super().__init__(model)

		self.N = N
		self.K = K
		self.fs = fs
		self.res_the = res_the
		self.res_phi = res_phi

		self.cat_maxCoor = cat_maxCoor
		self.apply_vad = apply_vad
		if apply_vad:
			self.vad = webrtcvad.Vad()
			self.vad.set_mode(3)

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform='PHAT')
		self.srp = at_modules.SRP_map(N, K, res_the, res_phi, rn, fs,
									  thetaMax=np.pi / 2 if arrayType == 'planar' else np.pi)

	def activate_vad(self, apply=True):
		self.apply_vad = apply
		if self.apply_vad:
			self.vad = webrtcvad.Vad()
			self.vad.set_mode(3)

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" Compute the SRP-PHAT maps from the microphone signals and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.unsqueeze(1) # Add channel axis

			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()

			maps = self.srp(self.gcc(mic_sig_batch))
			maximums = maps.view(list(maps.shape[:-2]) + [-1]).argmax(dim=-1)

			if self.cat_maxCoor:
				max_the = (maximums / self.res_phi).float() / maps.shape[-2]
				max_phi = (maximums % self.res_phi).float() / maps.shape[-1]
				repeat_factor = np.array(maps.shape)
				repeat_factor[:-2] = 1
				maps = torch.cat((maps,
								  max_the[..., None, None].repeat(repeat_factor.tolist()),
								  max_phi[..., None, None].repeat(repeat_factor.tolist())
								  ), 1)

			if self.apply_vad:
				if acoustic_scene_batch is not None:
					vad_batch =  np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
				assert vad_batch is not None # Breaks if neither acoustic_scene_batch nor vad_batch was given
				vad_output_th = vad_batch.mean(axis=2) > 2 / 3
				vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis]
				vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
				repeat_factor = np.array(maps.shape)
				repeat_factor[:-2] = 1
				maps *= vad_output_th.float().repeat(repeat_factor.tolist())

			output += [ maps ]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.tensor([acoustic_scene_batch[i].DOAw.astype(np.float32) for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [ DOAw_batch ]

		return output[0] if len(output)==1 else output


class OneSourceTrackingFromMaximumsLearner(OneSourceTrackingLearner):
	""" Learner for models which use the coordinates of the maximums of the SRP-PHAT maps as input
	"""
	def __init__(self, model, N, K, res_the, res_phi, rn, fs, c=343.0, arrayType='planar', apply_vad=False):
		"""
		model: Model to work with
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		res_the: Resolution of the maps in the elevation axis
		res_phi: Resolution of the maps in the azimuth axis
		rn: Position of each microphone relative to te center of the array
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		arrayType: 'planar' or '3D' whether all the microphones are in the same plane (and the maximum DOA elevation is pi/2) or not [default: 'planar']
		apply_vad: Turn to zero all the map pixels in frames without speech signal [default: False]
		"""
		super().__init__(model)

		self.N = N
		self.K = K
		self.res_the = res_the
		self.res_phi = res_phi

		self.apply_vad = apply_vad

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform='PHAT')
		self.srp = at_modules.SRP_map(N, K, res_the, res_phi, rn, fs,
									  thetaMax=np.pi / 2 if arrayType == 'planar' else np.pi)

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" Get the coordinates of the maximums of the SRP-PHAT maps and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.view((-1, 1) + tuple(mic_sig_batch.shape)[1:]) # Add channel dimension
			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()
			maps = self.srp(self.gcc(mic_sig_batch))
			maximums = maps.view([maps.shape[0], maps.shape[2], -1]).argmax(dim=-1)
			max_the = (maximums / self.res_phi).float() / maps.shape[-2]
			max_phi = (maximums % self.res_phi).float() / maps.shape[-1]
			x_batch = torch.stack((max_the, max_phi), dim=-1)
			x_batch.transpose_(1, 2)

			if self.apply_vad:
				if acoustic_scene_batch is not None:
					vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
				assert vad_batch is not None
				vad_output_th = vad_batch.mean(axis=2) > 2 / 3
				vad_output_th = vad_output_th[:, np.newaxis, :]
				vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(x_batch.device)
				x_batch *= vad_output_th.float()

			output += [ x_batch ]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.tensor([acoustic_scene_batch[i].DOAw.astype(np.float32) for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [ DOAw_batch ]

		return output[0] if len(output)==1 else output


class OneSourceTrackingFromGCCsLearner(OneSourceTrackingLearner):
	""" Learner for models which use the sequence of the Generalized Cross-Correlation functions as input
	"""
	def __init__(self, model, N, K, rn, fs, c=343.0, apply_vad=False):
		"""
		model: Model to work with
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		rn: Position of each microphone relative to te center of the array (to get the needed length of the GCC)
		fs: Sampling frequency
		c: Speed of the sound [default: 343.0]
		apply_vad: Turn to zero the GCCs in frames without speech signal [default: False]
		"""
		super().__init__(model)

		self.N = N
		self.K = K

		self.apply_vad = apply_vad

		self.nb_pairs = (self.N*(self.N-1))//2
		self.pair_idx = []
		for i in range(N):
			for j in range(i+1,N):
				self.pair_idx.append((i,j))

		dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
		self.tau_max = int(np.ceil(dist_max / c * fs))
		self.gcc = at_modules.GCC(N, K, tau_max=self.tau_max, transform='PHAT')

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" Get the GCC sequence and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.view((-1, 1) + tuple(mic_sig_batch.shape)[1:]) # Add channel dimension
			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()
			gccs_batch = self.gcc(mic_sig_batch)

			x_batch = torch.empty((gccs_batch.shape[0], gccs_batch.shape[2], self.nb_pairs, self.tau_max*2+1)).cuda()
			for i in range(self.nb_pairs):
				x_batch[:,:,i,:] = gccs_batch[:, 0, :, self.pair_idx[i][0], self.pair_idx[i][1], :]
			x_batch = x_batch.reshape((gccs_batch.shape[0], gccs_batch.shape[2], self.nb_pairs*(self.tau_max*2+1)))
			x_batch.transpose_(1,2)

			if self.apply_vad:
				if acoustic_scene_batch is not None:
					vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
				assert vad_batch is not None
				vad_output_th = vad_batch.mean(axis=2) > 2 / 3
				vad_output_th = vad_output_th[:, np.newaxis, :]
				vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(x_batch.device)
				x_batch *= vad_output_th.float()

			output += [x_batch]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.tensor([acoustic_scene_batch[i].DOAw.astype(np.float32) for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [ DOAw_batch ]

		return output[0] if len(output)==1 else output


class OneSourceTrackingSpectrogramLearner(OneSourceTrackingLearner):
	""" Learner for models which use the spectrogram of each microphone signalas input
	"""
	def __init__(self, model, N, K, apply_vad=False):
		"""
		model: Model to work with
		N: Number of microphones in the array
		K: Window size for the SRP-PHAT map computation
		apply_vad: Turn to zero all the frequencies in frames without speech signal [default: False]
		"""
		super().__init__(model)

		self.N = N
		self.K = K

		self.apply_vad = apply_vad

		self.nb_pairs = (self.N*(self.N-1))//2
		self.pair_idx = []
		for i in range(N):
			for j in range(i+1,N):
				self.pair_idx.append((i,j))

	def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
		""" Compute the spectrogram of each microphone signal and extract the DoA groundtruth from the AcousticScene
		"""
		output = []

		if mic_sig_batch is not None:
			mic_sig_batch = torch.from_numpy( mic_sig_batch.astype(np.float32) )
			mic_sig_batch = mic_sig_batch.view((-1, 1) + tuple(mic_sig_batch.shape)[1:]) # Add channel dimension
			if self.cuda_activated:
				mic_sig_batch = mic_sig_batch.cuda()

			mic_sig_fft = torch.rfft(mic_sig_batch, 1) # torch.Size([5, 1, 103, 12, 2049, 2])
			spect = at_modules.complex_cart2polar(mic_sig_fft)
			spect[..., 0] /= spect[..., 0].max(dim=4, keepdim=True)[0]
			spect[..., 1] /= np.pi
			x_batch = spect.permute(0,1,3,5,2,4).reshape((mic_sig_batch.shape[0], -1, mic_sig_batch.shape[2], mic_sig_batch.shape[-1]//2+1)) # torch.Size([5, 24, 103, 2049])
			x_batch = x_batch[..., 1:] # Remove f=0, torch.Size([5, 24, 103, 2048])

			if self.apply_vad:
				if acoustic_scene_batch is not None:
					vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
				assert vad_batch is not None
				vad_output_th = vad_batch.mean(axis=2) > 2 / 3
				vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis]
				vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(x_batch.device)
				x_batch *= vad_output_th.float()

			output += [x_batch]

		if acoustic_scene_batch is not None:
			DOAw_batch = torch.tensor([acoustic_scene_batch[i].DOAw.astype(np.float32) for i in range(len(acoustic_scene_batch))])
			if self.cuda_activated:
				DOAw_batch = DOAw_batch.cuda()
			output += [ DOAw_batch ]

		return output[0] if len(output)==1 else output