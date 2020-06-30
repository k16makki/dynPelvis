import glob
import os
import nibabel as nib
import numpy as np

opath = '/home/karimm/Bureau/Deep_project/AR_patho/simulated_distorted_sequence/'


basename= 'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'
dynamicSet = glob.glob(opath+basename+'*.nii.gz')
dynamicSet.sort()

print(dynamicSet)


#nii = nib.load(dynamicSet[0])
#a = nii.get_data()


#result = np.pad(a, ((130,130), (40,40), (0, 0)), 'constant')

#result = np.zeros((n,n,n))

#result[:a.shape[0],:a.shape[1],:a.shape[2]] = a

for t in range(len(dynamicSet)):
	print(t)

	nii = nib.load(dynamicSet[t])
	a = nii.get_data()

	result = np.pad(a, ((130,130), (40,40), (0, 0)), 'constant')	


	j = nib.Nifti1Image(result, nii.affine)
	#save_path2 = './test_zero_padding.nii.gz'
	nib.save(j, dynamicSet[t])





