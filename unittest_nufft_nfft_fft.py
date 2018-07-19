# coding: utf-8

from pysap.plugins.mri.reconstruct_3D.fourier import NUFFT, NFFT3, FFT3
import numpy as np
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D


# In[2]:
_mask = np.ones(np.random.randint(2, size=(64, 64, 64)).shape)
_samples = convert_mask_to_locations_3D(_mask)
_samples_shift = convert_mask_to_locations_3D(np.fft.fftshift(_mask))
image = np.load('/volatile/bsarthou/datas/NUFFT/mri_img_2D.npy')
image = image[64:128, 64:128]
images = np.tile(image, (64, 1, 1))


# In[3]:


fourier_op_dir_nufft = NUFFT(samples=_samples, platform='gpu',
                             shape=(64, 64, 64), Kd=64, Jd=1)
fourier_op_dir_nfft = NFFT3(samples=_samples, shape=(64, 64, 64))


# In[4]:


fourier_op_dir_fft = FFT3(samples=_samples_shift, shape=(64, 64, 64))


# In[20]:
kspace_nfft = fourier_op_dir_nfft.op(images)/np.sqrt(262144)
kspace_nufft = fourier_op_dir_nufft.op(images)
kspace_fft = np.fft.ifftshift(fourier_op_dir_fft.op(np.fft.fftshift(images)))
kspace_fft = kspace_fft.flatten()/np.sqrt(64*64*64)


# In[21]:


print(kspace_nufft.shape)
print(kspace_nfft.shape)
print(kspace_fft.shape)


# In[22]:


from modopt.math.metrics import mse

print(mse(kspace_nufft, kspace_nfft))
print(mse(kspace_nufft, kspace_fft))
print(mse(kspace_nfft, kspace_fft))
print(kspace_nfft.max())
print(kspace_fft.max())


# In[23]:


import matplotlib.pyplot as plt
# plt.subplot(311)
# plt.plot(np.abs(kspace_nufft))
# plt.subplot(312)
# plt.plot(np.abs(kspace_nfft))
# plt.subplot(313)
# plt.plot(np.abs(kspace_fft))
#
#
# # In[24]:
#
#
# plt.figure()
# plt.plot(np.abs(kspace_nufft - kspace_nfft))
# plt.figure()
# plt.plot(np.abs(kspace_nfft - kspace_fft))
# plt.figure()
# plt.plot(np.abs(kspace_nufft - kspace_fft))
# plt.show()

# # 3D plot
# from pysap.plugins.mri.reconstruct_3D.utils import gridding_3d
# print("regridding")
# k_space_fft_3D = gridding_3d(_samples_shift, kspace_fft, (64, 64, 64))
# print('1')
# k_space_nfft_3D = gridding_3d(_samples, kspace_nfft, (64, 64, 64))
# print('2')
# k_space_nufft_3D = gridding_3d(_samples, kspace_nufft, (64, 64, 64))
# print('regridding done')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
#
# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
#
# X, Y = meshgrid(np.arange(k_space_fft_3D.shape[0]),
#                 np.arange(k_space_fft_3D.shape[1]))
#
# ax1.plot_surface(X, Y, np.abs(k_space_fft_3D-k_space_nfft_3D),
#                  cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax2.plot_surface(X, Y, np.abs(k_space_fft_3D-k_space_nufft_3D),
#                  cmap=cm.coolwarm,
#                  linewidth=0, antialiased=False)
#
# plt.show()
# In[36]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.set_title('FFT vs. NFFT:\n l2={:4E}'.format(mse(kspace_fft, kspace_nfft)))
ax1.set_xlabel('Nb samples')
ax1.set_ylabel('Amplitude')
ax1.plot(np.abs(kspace_nfft - kspace_fft))

ax2.set_title('NFFT vs. NUFFT:\n l2={:4E}'.format(mse(kspace_nfft, kspace_nufft)))
ax2.set_xlabel('Nb samples')
ax2.set_ylabel('Amplitude')
ax2.plot(np.abs(kspace_nfft - kspace_nufft))

ax3.set_title('FFT vs. NUFFT:\n l2={:4E}'.format(mse(kspace_fft, kspace_nufft)))
ax3.set_xlabel('Nb samples')
ax3.set_ylabel('Amplitude')
plt.plot(np.abs(kspace_fft - kspace_nufft))
plt.tight_layout()


# In[10]:


non_z = np.nonzero(kspace_fft)
print(kspace_fft[non_z])
print(kspace_nfft[non_z])
print(kspace_nufft[non_z])


# In[11]:


kspace_input = np.copy(kspace_nfft)
fourier_op_adj_nufft = NUFFT(samples=_samples, platform='cpu', shape=(64,64,64), Kd=64, Jd=1)
fourier_op_adj_nfft = NFFT3(samples=_samples, shape=(64,64,64))
fourier_op_adj_fft = FFT3(samples=_samples_shift, shape=(64,64,64))


# In[12]:


img_nfft = fourier_op_adj_nfft.adj_op(kspace_input)
img_nufft = fourier_op_adj_nufft.adj_op(kspace_input)
#kspace_fft = kspace_fft.flatten()/np.sqrt(64*64*64)
kspace_input_fft = np.reshape(kspace_input, (64,64,64))
img_fft = np.fft.fftshift(fourier_op_adj_fft.adj_op(np.fft.ifftshift(kspace_input_fft)))


# In[43]:


print(mse(img_nufft, img_nfft))
print(mse(img_nufft, img_fft))
print(mse(img_nfft, img_fft))


stop
# In[14]:


plt.figure()
plt.subplot(121)
plt.imshow(np.abs(img_nfft[:,:,10]),cmap = 'gray')
plt.subplot(122)
plt.imshow(np.abs(img_fft[:,:,10]), cmap='gray')


# In[42]:


_min, _max = np.min(np.abs(np.abs(img_nfft[:,:,10]) - np.abs(img_nufft[:,:,10]))), np.max(np.abs(np.abs(img_nfft[:,:,10]) - np.abs(img_nufft[:,:,10])))

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title('FFT vs. NFFT:\n l2={:4E}'.format(mse(np.abs(img_fft[:,:,10]), np.abs(img_nfft[:,:,10]))))
ax1.set_xlabel('Nb samples')
ax1.set_ylabel('Amplitude')
ax1.imshow(np.abs(np.abs(img_nfft[:,:,10]) - np.abs(img_fft[:,:,10])), vmin=_min, vmax=_max)

ax2.set_title('NFFT vs. NUFFT:\n l2={:4E}'.format(mse(np.abs(img_nfft[:,:,10]), np.abs(img_nufft[:,:,10]))))
ax2.set_xlabel('Nb samples')
ax2.set_ylabel('Amplitude')
ax2.imshow(np.abs(np.abs(img_nfft[:,:,10]) - np.abs(img_nufft[:,:,10])), vmin=_min, vmax=_max)

ax3.set_title('FFT vs. NUFFT:\n l2={:4E}'.format(mse(np.abs(img_fft[:,:,10]), np.abs(img_nufft[:,:,10]))))
ax3.set_xlabel('Nb samples')
ax3.set_ylabel('Amplitude')
plt.imshow(np.abs(np.abs(img_nfft[:,:,10]) - np.abs(img_nufft[:,:,10])), vmin=_min, vmax=_max)
plt.tight_layout()
plt.show()
