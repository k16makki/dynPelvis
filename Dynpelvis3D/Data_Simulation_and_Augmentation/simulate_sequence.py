# -*- coding: utf-8 -*-

"""
  © Aix Marseille University - LIS-CNRS UMR 7020
  Author(s): Karim Makki (karim.makki@univ-amu.fr)
  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

import glob
import os
import nibabel as nib
import numpy as np

opath = '/home/karimm/Bureau/Deep_project/AR_patho/simulated_distorted_sequence/'

if not os.path.exists(opath):
        os.makedirs(opath)

input_shape = '/home/karimm/Bureau/Deep_project/AR/AR_Anat3D_T1-TFI-Expi_swapDim_mask.nii.gz'

go0 ='time python Exponential_map.py -in '+ input_shape +' -refweight /home/karimm/Bureau/Deep_project/AR_patho/S1.nii.gz -refweight /home/karimm/Bureau/Deep_project/AR_patho/S2.nii.gz -refweight /home/karimm/Bureau/Deep_project/AR_patho/S3.nii.gz -refweight /home/karimm/Bureau/Deep_project/AR_patho/S4.nii.gz -t /home/karimm/Bureau/Deep_project/AR_patho/T0.mat -t /home/karimm/Bureau/Deep_project/AR_patho/T1.mat -t /home/karimm/Bureau/Deep_project/AR_patho/T2.mat -t /home/karimm/Bureau/Deep_project/AR_patho/T3.mat -o '+ opath +  ' -expmap 0 '

l = 30

for t in range(30):
	warped_image = 'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'+"{:04d}".format(t+1)+'.nii.gz'
	def_field = 'Deformation_field_'+"{:04d}".format(t+1)+'.nii.gz'
	temp_interp = (t+1)/l
	go = go0+ ' -warped_image '+ warped_image + ' -def_field '+ def_field + ' -tempinterp ' +  str(temp_interp)
	print(go)
	os.system(go)


basename= 'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'
dynamicSet = glob.glob(opath+basename+'*.nii.gz')
dynamicSet.sort()


bin0 = 'fsl5.0-fslmaths '

for t in range (len(dynamicSet)):
	binarize = bin0+ dynamicSet[t] + ' -thr 0.4 -bin '+ dynamicSet[t]
	print(binarize)
	os.system(binarize)


###To complete one motion cycle: from max inspi to rest.state

## add input image as first image in the sequence

go = 'cp '+  input_shape + ' ' + opath+'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_0000.nii.gz'

print(go)
os.system(go)


for t in range (1,len(dynamicSet)):
	#print(dynamicSet[l-t-1])
	#a = "{:04d}".format(t+l)

	new_image = opath+'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'+"{:04d}".format(t+l)+'.nii.gz'
	go = 'cp '+  dynamicSet[l-t-1] + ' ' + new_image
	print(go)
	os.system(go)

#print("{:04d}".format(2*l))

'''
go = 'cp '+  input_shape + ' ' + opath+'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'+"{:04d}".format(2*l)+'.nii.gz'

print(go)
os.system(go)	
'''

