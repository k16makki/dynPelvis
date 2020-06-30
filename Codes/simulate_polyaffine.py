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


import numpy as np
import nibabel as nib
import math


subject_name = 'AR_patho'

dynamic_volume ='/home/karimm/Bureau/Deep_project/AR/AR_Anat3D_T1-TFI-Expi_swapDim_mask.nii.gz'

nii = nib.load(dynamic_volume)

nx, ny, nz =  nii.get_data().shape


coords = np.zeros((3,nx, ny, nz), dtype='float32')
coords[0,...] = np.arange(nx)[:,np.newaxis,np.newaxis]
coords[1,...] = np.arange(ny)[np.newaxis,:,np.newaxis]
coords[2,...] = np.arange(nz)[np.newaxis,np.newaxis,:]


def Matrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ', fmt='%1.3f')


def simulate_affine_matrix(translation=None, angles=None, scale=None, shear=None):
  """
  compute the affine matrix using a direct computation
  faster computation than numpy matrix multiplication
  """
  
  mat = np.identity(4)
  gx,gy,gz = 0.0,0.0,0.0
  sx,sy,sz = 1.0,1.0,1.0
  if translation is not None:
    mat[:3,3] = translation[:3]
  if angles is not None:
    ax = math.pi*angles[0]/180.0
    ay = math.pi*angles[1]/180.0
    az = math.pi*angles[2]/180.0
    cosx = math.cos(ax)
    cosy = math.cos(ay)
    cosz = math.cos(az)
    sinx = math.sin(ax)
    siny = math.sin(ay)
    sinz = math.sin(az)
  if shear is not None:
    gx = shear[0]
    gy = shear[1]
    gz = shear[2]
  if scale is not None:
    sx = scale[0]
    sy = scale[1]
    sz = scale[2]
    
  mat[0,0] = sx * cosy * (cosz + (gy*sinz) )
  mat[0,1] = sy * (cosy * (sinz + (gx * gy * cosz)) - (gz * siny) )
  mat[0,2] = sz * ( (gx * cosy * cosz) - siny)
  mat[1,0] = sx * (sinx * siny * (cosz + gy * sinz) - cosx * (sinz + (gy * cosz) ))
  mat[1,1] = sy * (sinx * siny * (sinz + (gx * gz * cosz) ) + cosx * (cosz - (gx * gy * sinz)) + (gz * sinx * cosy))
  mat[1,2] = sz * (sinx * cosy + (gx * (sinx * siny * cosz - cosx * sinz)))
  mat[2,0] = sx * (cosx * siny * (cosz + (gy * sinz)) + sinx * (sinz - (gy * cosz) ))
  mat[2,1] = sy * (cosx * siny * (sinz + (gx * gz * cosz)) - sinx * (cosz - (gx * gz * sinz)) + (gz * cosx * cosy) )
  mat[2,2] = sz * (cosx * cosy + (gx * ( (cosx * siny * cosz) + (sinx * sinz) )) )
  
  return np.round(mat, 3)

def simulate_sphere(center, radius):


	
	sphere = np.where( (np.power(coords[0,...]-center[0],2) + np.power(coords[1,...]-center[1],2) + np.power(coords[2,...]-center[2],2)) <=radius**2)
      
	return sphere[0], sphere[1], sphere[2]

radius = 8

sph = simulate_sphere([37,178,150], radius)

S1 = np.zeros((nx, ny, nz))

S1[sph[0], sph[1], sph[2]] = 1

i = nib.Nifti1Image(S1, nii.affine)
nib.save(i, './'+subject_name+'/S1.nii.gz')


sph1 = simulate_sphere([37,155,156], radius)

S2 = np.zeros((nx, ny, nz))

S2[sph1[0], sph1[1], sph1[2]] = 1

i = nib.Nifti1Image(S2, nii.affine)
nib.save(i, './'+subject_name+'/S2.nii.gz')

sph2 = simulate_sphere([37,171,198], radius)

S3 = np.zeros((nx, ny, nz))

S3[sph2[0], sph2[1], sph2[2]] = 1

i = nib.Nifti1Image(S3, nii.affine)
nib.save(i, './'+subject_name+'/S3.nii.gz')


sph3 = simulate_sphere([37,130,188], radius)
S4 = np.zeros((nx, ny, nz))

S4[sph3[0], sph3[1], sph3[2]] = 1

i = nib.Nifti1Image(S4, nii.affine)
nib.save(i, './'+subject_name+'/S4.nii.gz')




m1 = simulate_affine_matrix(translation=(0,0,0), angles=(0,0,0), scale=(1,1.2,1.2), shear=None)

Matrix_to_text_file(m1, './'+subject_name+'/T0.mat')

m2 = simulate_affine_matrix(translation=(0,0,0), angles=(0,0,3), scale=(1,1,1.1), shear=None)

Matrix_to_text_file(m2, './'+subject_name+'/T1.mat')

m3 = simulate_affine_matrix(translation=(0,0,0), angles=(0,0,-3), scale=(1,1,1), shear=None)

Matrix_to_text_file(m3, './'+subject_name+'/T2.mat')


m4 = simulate_affine_matrix(translation=(0,0,0), angles=(0,0,0), scale=(1,1,1), shear=None)

Matrix_to_text_file(m4, './'+subject_name+'/T3.mat')

print(m4)
