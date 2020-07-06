# -*- coding: utf-8 -*-

"""
  © Aix Marseille University - LIS-CNRS UMR 7020
  Author(s): Karim Makki, Amine Bohi (karim.makki, amine bohi{@univ-amu.fr})
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
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy import ndimage
import glob
import os


#image = '/home/karim/Bureau/Karim/TV_Dyn3D_5SL_3dRecbyRegContours_Filled/TV_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_0004.nii.gz'
#tracked_points = '/home/karim/Bureau/input_data/TV_Dyn3D_5SL/output_step10/output_TV_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_0004/DeterministicAtlas__Reconstruction__bladder__subject_subj1.vtk'


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())


def vtk_to_array(filename):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    array = points.GetData()
    return vtk_to_numpy(array)



### Determine the nifti dynamic sequence
subject_name = 'AF_Dyn3D_5SL' #### To be modified

PATH_GLOBAL = "/home/karimm/Bureau/Karim"
sequence_name= subject_name+'_3dRecbyRegContours_Filled' 
sequence_path = PATH_GLOBAL + '/' + sequence_name
nifti_dynamicSet = glob.glob(sequence_path+'/'+'*.nii.gz')
nifti_dynamicSet.sort()

print(nifti_dynamicSet)


vtk_pointset_path = '/home/karimm/Bureau/input_data/'+subject_name+'/output_step10' 
vtk_pointset_sequence = glob.glob(vtk_pointset_path+'/*')
vtk_pointset_sequence.sort()

output_path = '/home/karimm/Bureau/sphere_projection/evaluation'

### Determine the initial point cloud or the pointset being tracked_points

PC0 = '/home/karimm/Bureau/input_data/points_M0_vtk/step10/'+subject_name+'_3dRecbyReg-Expi_FilledContour_0000.vtk'
pts0 = vtk_to_array(PC0)


for t in range(1,len(vtk_pointset_sequence)):

    print(t)


    prefix = nifti_dynamicSet[t].split('/')[-1].split('.')[0]

    tracked_pointset_vtk = vtk_pointset_sequence[t]+'/DeterministicAtlas__Reconstruction__bladder__subject_subj1.vtk'
    points = vtk_to_array(tracked_pointset_vtk)

    result_path = output_path+'/'+sequence_name
    if not os.path.exists(result_path):
           os.makedirs(result_path)
    resulting_file = result_path + '/'+prefix+'.nii.gz'


    #points = vtk_to_array(tracked_points)

    img = nifti_to_array(nifti_dynamicSet[t])

    # Determine organ centroid (i.e. sphere center)
    x, y, z = np.nonzero(img)
    mesh_centroid=[int(np.median(x)), int(np.median(y)),int(np.median(z))]

    # Draw the sphere (outer_boundary): Compute coordinates in the input image
    nx, ny, nz = img.shape
    coords = np.zeros((3,nx, ny, nz), dtype='float32')
    coords[0,...] = np.arange(nx)[:,np.newaxis,np.newaxis]
    coords[1,...] = np.arange(ny)[np.newaxis,:,np.newaxis]
    coords[2,...] = np.arange(nz)[np.newaxis,np.newaxis,:]

    # Sphere radius

    r=50

    # Define the outer boundary
    sphere = np.where( (np.power(coords[0,...]-mesh_centroid[0],2) + np.power(coords[1,...]-mesh_centroid[1],2) + np.power(coords[2,...]-mesh_centroid[2],2)) <=r**2 )
    out_boundary = np.ones(img.shape)
    out_boundary[sphere[0], sphere[1], sphere[2]] = 0

    # Define the inner boundary: the organ

    ## To avoid out of range problem in the borders during max_iterations
    img[...,0] = 0
    img[...,nz-1] = 0
    inner_boundary = img

    hdr = nib.load(nifti_dynamicSet[t]).header # Anatomical image header: this will give the voxel spacing along each axis (dx, dy, dz)
    dx = hdr.get_zooms()[0] #x-voxel spacing
    dy = hdr.get_zooms()[1] #y-voxel spacing
    dz = hdr.get_zooms()[2] #z-voxel spacing

    u = np.zeros((nx,ny,nz))  #u(i)
    u_n = np.zeros((nx,ny,nz)) #u(i+1)
            #
    L0 = np.zeros((nx,ny,nz))
    L1 = np.zeros((nx,ny,nz))

    L0_n = np.zeros((nx,ny,nz))
    L1_n = np.zeros((nx,ny,nz))

    Nx = np.zeros((nx,ny,nz))
    Ny = np.zeros((nx,ny,nz))
    Nz = np.zeros((nx,ny,nz))

    thickness = np.zeros((nx,ny,nz))

    #Dirichlet boundary conditions

    ## Binary erosion of the organ (3 voxels)

    kernel_width = 5

    img = ndimage.binary_erosion(img, structure=np.ones((kernel_width,kernel_width,kernel_width)) ).astype(img.dtype)

    in_BC = np.where(img!=0)

    #in_BC =	mesh_centroid

    out_BC = np.where(out_boundary!=0)

    #Initial conditions
    u[out_BC]= 100
    u[in_BC]= 0

    # Boundary conditions for computing L0 and L1

    L0[:,:,:]= -(dx+dy+dz)/6
    L1[:,:,:]= -(dx+dy+dz)/6

    # Solving the Laplace equation using the Jacobi iterative method

    # determine the region inside which the thickness is to be measured
    R = np.where(out_boundary+img == 0)


    n_iter = 300

    for it in range (n_iter):

        u_n = u

        u[R[0],R[1],R[2]]= (u_n[R[0]+1,R[1],R[2]] + u_n[R[0]-1,R[1],R[2]] +  u_n[R[0],R[1]+1,R[2]] \
        + u_n[R[0],R[1]-1,R[2]] + u_n[R[0],R[1],R[2]+1] + u_n[R[0],R[1],R[2]-1]) / 6
        del u_n


    print("Laplace's equation is solved")

    ##Compute the normalized tangent vector field of the correspondence trajectories

    N_xx, N_yy, N_zz = np.gradient(u)
    grad_norm = np.sqrt(N_xx**2 + N_yy**2 + N_zz**2)
    grad_norm[np.where(grad_norm==0)] = 1 ## to avoid dividing by zero

    del u

    # Normalization
    np.divide(N_xx, grad_norm, Nx)
    np.divide(N_yy, grad_norm, Ny)
    np.divide(N_zz, grad_norm, Nz)
    #gaussian_filter(Nx, sigma=1, output=Nx)
    #gaussian_filter(Ny, sigma=1, output=Ny)
    #gaussian_filter(Nz, sigma=1, output=Nz)

    del grad_norm, N_xx, N_yy, N_zz
    print("The normalized tangent vector field is successfully computed")

    den = np.absolute(Nx)+ np.absolute(Ny) + np.absolute(Nz)

    den[np.where(den==0)] = 1 ## to avoid dividing by zero

    # iteratively compute correspondence trajectory lengths L0 and L1

    for it in range (100):

         L0_n = L0
         L1_n = L1

         L0[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L0_n[(R[0]-np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
         np.absolute(Ny[R[0],R[1],R[2]]) * L0_n[R[0],(R[1]-np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
         + np.absolute(Nz[R[0],R[1],R[2]]) * L0_n[R[0],R[1],(R[2]-np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]

         L1[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L1_n[(R[0]+np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
         np.absolute(Ny[R[0],R[1],R[2]]) * L1_n[R[0],(R[1]+np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
         + np.absolute(Nz[R[0],R[1],R[2]]) * L1_n[R[0],R[1],(R[2]+np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]


         del L0_n, L1_n


    # compute  the thickness of the tissue region inside R

    thickness[R[0],R[1],R[2]] = L0[R[0],R[1],R[2]] + L1[R[0],R[1],R[2]]

    del L0, L1

    print("Mean thickness inside R:\n")
    print(np.mean(thickness[R[0],R[1],R[2]]))
    print("Maximum thickness inside R:\n")
    print(np.max(thickness[R[0],R[1],R[2]]))
    print("Minimum thickness inside R:\n")
    print(np.min(thickness[R[0],R[1],R[2]]))

    pts = np.zeros(img.shape)

    pts[pts0[:,0].astype(int), pts0[:,1].astype(int), pts0[:,2].astype(int)] = thickness[points[:,0].astype(int), points[:,1].astype(int), points[:,2].astype(int)]

    del thickness


    # m = nib.Nifti1Image(thickness, nib.load(image).affine)
    # outpath = '/home/karim/Bureau/thickk_samedi.nii.gz'
    # nib.save(m, outpath)

    n = nib.Nifti1Image(pts, nib.load(nifti_dynamicSet[t]).affine)
    #outpath = '/home/karim/Bureau/pts_samedi.nii.gz'
    nib.save(n, resulting_file)
