"""
	Pytorch models for DOA estimation.

	File name: acousticTrackingModels.py
	Author: David Diaz-Guerra
	Date creation: 05/2020
	Python Version: 3.8
	Pytorch Version: 1.4.0
"""

import torch
import torch.nn as nn

import acousticTrackingModules as at_modules


class Cross3D(nn.Module):
	""" Proposed model with causal 3D convolutions and two branches with different pooling in each axis SRP-PHAT map axis.
	"""
	def __init__(self, res_the, res_phi, in_deep=1, in_krnl_sz=(5, 5, 5), in_nb_ch=32, pool_sz=(1, 1, 1),
				 cr_deep=4, crThe_krnl_sz=(5, 3, 3), crPhi_krnl_sz=(5, 3, 3), cr_nb_ch=32, crThe_pool_sz=(1, 1, 2),
				 crPhi_pool_sz=(1, 2, 1), out_conv_len=5, out_conv_dilation=2, out_nbh=128):
		"""
		res_the: elevation resolution of the input maps
		res_phi: azimuth resolution of the input maps
		in_deep: Number of convolutional layers at the input [default: 1]
		in_krnl_sz: Kernel size of the convolutional layers at the input [default: (5,5,5)]
		in_nb_ch: Number of channels of the convolutional layers at the input [default: 32]
		pool_sz: Kernel size and stride of the max pooling layer after the initial 3D CNNs [default: (1,1,1)]
		cr_deep: Number of convolutional layers to apply in cross branches [default: 4]
		crThe_krnl_sz: Kernel size of the convolutional layers at the theta branch of the cross [default: (5,3,3)]
		crPhi_krnl_sz: Kernel size of the convolutional layers at the phi branch of the cross [default: (5,3,3)]
		cr_nb_ch: Number of channels of the convolutional layers at the cross branches [default: 32]
		crThe_pool_sz: Kernel size and stride of the max pooling layer between each convolutional layer of the theta branch [default: (1,1,2)]
		crPhi_pool_sz: Kernel size and stride of the max pooling layer between each convolutional layer of the theta branch [default: (1,2,1)]
		out_conv_len: Kernel size of the two 1D convolutional layers at end of the network [default: 5]
		out_conv_dilation: Dilation of the two 1D convolutional layers at end of the network [default: 2]
		out_nbh: Number of channels of the first of the two 1D convolutional layers at end of the network [default: 128]
		"""

		super(Cross3D, self).__init__()

		self.res_the = res_the
		self.res_phi = res_phi
		self.in_deep = in_deep
		self.cr_deep = cr_deep
		self.in_nb_ch = in_nb_ch
		self.cr_nb_ch = cr_nb_ch

		self.crThe_resThe = res_the // pool_sz[1] // crThe_pool_sz[1] ** cr_deep
		self.crThe_resPhi = res_phi // pool_sz[2] // crThe_pool_sz[2] ** cr_deep
		self.crPhi_resThe = res_the // pool_sz[1] // crPhi_pool_sz[1] ** cr_deep
		self.crPhi_resPhi = res_phi // pool_sz[2] // crPhi_pool_sz[2] ** cr_deep
		self.crThe_nb_outAct = self.crThe_resThe * self.crThe_resPhi * cr_nb_ch
		self.crPhi_nb_outAct = self.crPhi_resThe * self.crPhi_resPhi * cr_nb_ch

		self.in_sphPad = at_modules.SphericPad((in_krnl_sz[2] // 2,) * 2 + (in_krnl_sz[1] // 2,) * 2)
		self.in_conv = nn.ModuleList([at_modules.CausConv3d(3, in_nb_ch, in_krnl_sz)] if in_deep > 0 else [])
		self.in_conv += nn.ModuleList(
			[at_modules.CausConv3d(in_nb_ch, in_nb_ch, in_krnl_sz) for i in range(in_deep - 1)])
		self.in_prelu = nn.ModuleList([nn.PReLU(in_nb_ch) for i in range(in_deep)])

		self.pool = nn.MaxPool3d(pool_sz)

		self.crThe_sphPad = at_modules.SphericPad((crThe_krnl_sz[2] // 2,) * 2 + (crThe_krnl_sz[1] // 2,) * 2)
		self.crThe_conv = nn.ModuleList(
			[at_modules.CausConv3d(in_nb_ch, cr_nb_ch, crThe_krnl_sz)] if cr_deep > 0 else [])
		self.crThe_conv += nn.ModuleList(
			[at_modules.CausConv3d(cr_nb_ch, cr_nb_ch, crThe_krnl_sz) for i in range(cr_deep - 1)])
		self.crThe_prelu = nn.ModuleList([nn.PReLU(cr_nb_ch) for i in range(cr_deep)])
		self.crThe_pool = nn.MaxPool3d(crThe_pool_sz)

		self.crPhi_sphPad = at_modules.SphericPad((crPhi_krnl_sz[2] // 2,) * 2 + (crPhi_krnl_sz[1] // 2,) * 2)
		self.crPhi_conv = nn.ModuleList(
			[at_modules.CausConv3d(in_nb_ch, cr_nb_ch, crPhi_krnl_sz)] if cr_deep > 0 else [])
		self.crPhi_conv += nn.ModuleList(
			[at_modules.CausConv3d(cr_nb_ch, cr_nb_ch, crPhi_krnl_sz) for i in range(cr_deep - 1)])
		self.crPhi_prelu = nn.ModuleList([nn.PReLU(cr_nb_ch) for i in range(cr_deep)])
		self.crPhi_pool = nn.MaxPool3d(crPhi_pool_sz)

		self.out_conv1 = at_modules.CausConv1d(self.crThe_nb_outAct + self.crPhi_nb_outAct, out_nbh, out_conv_len, dilation=out_conv_dilation)
		self.out_prelu = nn.PReLU()
		self.out_conv2 = at_modules.CausConv1d(out_nbh, 3, out_conv_len, dilation=out_conv_dilation)

	def forward(self, x):
		for i in range(self.in_deep):
			x = self.in_prelu[i](self.in_conv[i](self.in_sphPad(x)))
		x = self.pool(x)

		xThe = x
		xPhi = x
		for i in range(self.cr_deep):
			xThe = self.crThe_prelu[i](self.crThe_pool(self.crThe_conv[i](self.crThe_sphPad(xThe))))
			xPhi = self.crPhi_prelu[i](self.crPhi_pool(self.crPhi_conv[i](self.crPhi_sphPad(xPhi))))

		xThe = xThe.transpose(1, 2).contiguous().view(-1, x.shape[-3], self.crThe_nb_outAct)
		xPhi = xPhi.transpose(1, 2).contiguous().view(-1, x.shape[-3], self.crPhi_nb_outAct)

		x = torch.cat((xThe, xPhi), dim=2)
		x = x.transpose(1, 2)

		x = self.out_prelu(self.out_conv1(x))
		x = torch.tanh((self.out_conv2(x)))
		x = x.transpose(1, 2)

		return x


class Cnn1D(nn.Module):
	""" Causal 1D Convolutional Neural Network to use over SRP-PHAT maximums or GCCs sequences.
	"""
	def __init__(self, input_ch, in_deep=1, in_krnl_sz=5, in_nb_ch=1024,
				 md_deep=4, md_krnl_sz=5, md_nb_ch=512,
				 out_conv_len=5, out_conv_dilation=2, out_nbh=128):
		"""
		input_ch: Number of input sequences
		in_deep: Number of convolutional layers in the first set [default: 1]
		in_krnl_sz: Kernel size of the convolutional layers of the first set [default: 5]
		in_nb_ch: Number of channels of the convolutional layers of the first set [default: 1024]
		md_deep: Number of convolutional layers in the second set [default: 4]
		md_krnl_sz: Kernel size of the convolutional layers of the second set [default: 5]
		md_nb_ch: Number of channels of the convolutional layers of the second set [default: 2]
		out_conv_len: Kernel size of the two convolutional layers at end of the network [default: 5]
		out_conv_dilation: Dilation of the two 1D convolutional layers at end of the network [default: 2]
		out_nbh: Number of channels of the first of the two 1D convolutional layers at end of the network [default: 128]
		"""

		super(Cnn1D, self).__init__()

		self.in_deep = in_deep
		self.md_deep = md_deep
		self.in_nb_ch = in_nb_ch
		self.md_nb_ch = md_nb_ch

		self.in_conv = nn.ModuleList([at_modules.CausConv1d(input_ch, in_nb_ch, in_krnl_sz)] if in_deep > 0 else [])
		self.in_conv += nn.ModuleList(
			[at_modules.CausConv1d(in_nb_ch, in_nb_ch, in_krnl_sz) for i in range(in_deep - 1)])
		self.in_prelu = nn.ModuleList([nn.PReLU(in_nb_ch) for i in range(in_deep)])

		self.md_conv = nn.ModuleList(
			[at_modules.CausConv1d(in_nb_ch, md_nb_ch, md_krnl_sz)] if md_deep > 0 else [])
		self.md_conv += nn.ModuleList(
			[at_modules.CausConv1d(md_nb_ch, md_nb_ch, md_krnl_sz) for i in range(md_deep - 1)])
		self.md_prelu = nn.ModuleList([nn.PReLU(md_nb_ch) for i in range(md_deep)])

		self.out_conv1 = at_modules.CausConv1d(md_nb_ch, out_nbh, out_conv_len, dilation=out_conv_dilation)
		self.out_prelu = nn.PReLU()
		self.out_conv2 = at_modules.CausConv1d(out_nbh, 3, out_conv_len, dilation=out_conv_dilation)

	def forward(self, x):
		for i in range(self.in_deep):
			x = self.in_prelu[i](self.in_conv[i](x))

		for i in range(self.md_deep):
			x = self.md_prelu[i](self.md_conv[i](x))

		x = self.out_prelu(self.out_conv1(x))
		x = torch.tanh((self.out_conv2(x)))
		x = x.transpose(1, 2)
		return x


class Cnn2D(nn.Module):
	""" Causal 2D Convolutional Neural Network to use over spectrograms.
	"""
	def __init__(self, nb_freqs, nb_in_channels, cnn_deep=5, cnn_nb_ch=128, cnn_krnl_sz=(5, 5), pool_sz=(1, 4),
				 out_conv_len=5, out_conv_dilation=2, out_nbh=128):
		"""
		nb_freqs: Number of frequencies of the spectrograms
		nb_in_channels: Number of input channels (one spectrogram per microphone)
		cnn_deep: Number of 2D convolutional layers [default: 5]
		cnn_nb_ch: Number of channels of the 2D convolutional layers [default: 128]
		cnn_krnl_sz: Kernel size of the 2D convolutional layers [default: (5,5)]
		pool_sz: Kernel size and stride of the max poolings performed after each 2D convolutional layer [default: (1,4)]
		out_conv_len: Kernel size of the two convolutional layers at end of the network [default: 5]
		out_conv_dilation: Dilation of the two 1D convolutional layers at end of the network [default: 2]
		out_nbh: Number of channels of the first of the two 1D convolutional layers at end of the network [default: 128]
		"""

		super(Cnn2D, self).__init__()

		self.nb_freqs = nb_freqs
		self.nb_in_channels = nb_in_channels
		self.cnn_deep = cnn_deep
		self.cnn_nb_ch = cnn_nb_ch

		self.out_cnn_nb_freqs = nb_freqs // pool_sz[1] ** cnn_deep

		self.pad = nn.ReflectionPad2d((cnn_krnl_sz[1] // 2,) * 2 + (0, 0))
		self.cnn_conv = nn.ModuleList([at_modules.CausConv2d(nb_in_channels, cnn_nb_ch, cnn_krnl_sz)] if cnn_deep > 0 else [])
		self.cnn_conv += nn.ModuleList(
			[at_modules.CausConv2d(cnn_nb_ch, cnn_nb_ch, cnn_krnl_sz) for i in range(cnn_deep - 1)])
		self.in_prelu = nn.ModuleList([nn.PReLU(cnn_nb_ch) for i in range(cnn_deep)])

		self.pool = nn.MaxPool2d(pool_sz)

		self.out_conv1 = at_modules.CausConv1d(self.out_cnn_nb_freqs * cnn_nb_ch, out_nbh, out_conv_len, dilation=out_conv_dilation)
		self.out_prelu = nn.PReLU()
		# self.out_conv2 = at_modules.CausConv1d(out_nbh, 2, out_conv_len)
		self.out_conv2 = at_modules.CausConv1d(out_nbh, 3, out_conv_len, dilation=out_conv_dilation)

		# self.denormalize = at_modules.DeNormalizeCoord((0, theta_max), (-np.pi, np.pi))

	def forward(self, x):
		for i in range(self.cnn_deep):
			x = self.in_prelu[i](self.cnn_conv[i](self.pad(x)))
			x = self.pool(x)

		x = x.transpose(2, 3)
		x = x.reshape(x.shape[0], self.cnn_nb_ch * self.out_cnn_nb_freqs, -1)

		#x = self.dropout(x)
		x = self.out_prelu(self.out_conv1(x))
		#x = self.dropout(x)
		x = torch.tanh((self.out_conv2(x)))
		x = x.transpose(1, 2)

		# x = self.denormalize(x)
		return x


class SELDnet(nn.Module):
	""" SELDnet architecture simplified to perform only the DOA estimation of one source.
	Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen, "Sound event localization and detection
	of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal
	Processing (JSTSP 2018) https://arxiv.org/pdf/1807.00129.pdf
	"""
	def __init__(self, C, P=64, MP=(8,8,2), Q=128, R=128):
		super(SELDnet, self).__init__()
		self.P = P

		self.cnn_conv = nn.ModuleList([nn.Conv2d(2*C, P, (3,3), padding=1)])
		self.cnn_conv += nn.ModuleList([nn.Conv2d(P, P, (3,3), padding=1) for i in range(3-1)])
		self.pool = nn.ModuleList([nn.MaxPool2d((1,MP[i])) for i in range(3)])

		self.gru = nn.GRU(input_size=2*P, hidden_size=Q//2, num_layers=2, bidirectional=True, batch_first=True)

		self.fc1 = nn.Linear(Q, R)
		self.fc2 = nn.Linear(R, 3)

	def forward(self, x):
		for i in range(len(self.cnn_conv)):
			x = nn.functional.relu(self.cnn_conv[i](x))
			x = self.pool[i](x)

		x = x.transpose(1, 2)
		x = x.reshape(x.shape[0], -1, 2*self.P)
		x = self.gru(x)[0]

		x = self.fc1(x) # SIC
		x = self.fc2(x)

		return x
