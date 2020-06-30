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
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy import ndimage
import glob
import os
import open3d as o3d
from mayavi.mlab import *
import mayavi
from scipy.ndimage.filters import gaussian_filter
import argparse



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

def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())




def extract_contour(filled_form):

    img = nib.load(filled_form).get_data()
    distance_map = ndimage.distance_transform_edt(img)
    contour = np.zeros(img.shape)
    contour[np.where(distance_map==1)]=1

    return contour

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--shape', help='3D binary mask of the shape', type=str, required = True)
    #parser.add_argument('-t', '--time', help='time index', type=str, required = True)
    parser.add_argument('-opath', '--output', help='output path', type=str)
    parser.add_argument('-pts', '--points', help='set of contour points, as vtk file', type=str)
    parser.add_argument('-ker', '--kernel', help='erosion kernel', type=int, default=7)
    parser.add_argument('-r', '--radius', help='radius of the surrounding sphere', type=float, default=60)
    parser.add_argument('-s', '--sampling', help='sampling rate, relative to voxel size', type=float, default=2)



    args = parser.parse_args()

    opath = '.'
    if args.output is not None :
        opath = args.output

    output = opath+'/feature'
    if not os.path.exists(output):
        os.makedirs(output)


    output_image = extract_contour(args.shape)
    binary_mask = np.where(output_image!=0)
    centroid = [np.mean(binary_mask[0]),np.mean(binary_mask[1]),np.mean(binary_mask[2])]
    nx, ny, nz = nib.load(args.shape).get_data().shape ## image dimensions

    points = np.zeros((len(binary_mask[0]),3))
    points[:,0] = binary_mask[0]
    points[:,1] = binary_mask[1]
    points[:,2] = binary_mask[2]


    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.voxel_down_sample(voxel_size=args.sampling)
    #o3d.visualization.draw_geometries([downpcd])
    points = np.asarray(downpcd.points)

    if args.points is not None :
        points = vtk_to_array(args.points)


    coords = np.zeros((3,nx, ny, nz), dtype='float32')
    coords[0,...] = np.arange(nx)[:,np.newaxis,np.newaxis]
    coords[1,...] = np.arange(ny)[np.newaxis,:,np.newaxis]
    coords[2,...] = np.arange(nz)[np.newaxis,np.newaxis,:]

    # Define the outer boundary
    sphere = np.where( (np.power(coords[0,...]-int(centroid[0]),2) + np.power(coords[1,...]-int(centroid[1]),2) + np.power(coords[2,...]-int(centroid[2]),2)) <=args.radius**2 )
    out_boundary = np.ones((nx,ny,nz))
    out_boundary[sphere[0], sphere[1], sphere[2]] = 0

    # Define the inner boundary: the organ

    img = nifti_to_array(args.shape)
    ## To avoid out of range problem in the borders during max_iterations
    img[...,0] = 0
    img[...,nz-1] = 0

    hdr = nib.load(args.shape).header # Anatomical image header: this will give the voxel spacing along each axis (dx, dy, dz)
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

    ## Binary erosion of the shape

    img = ndimage.binary_erosion(img, structure=np.ones((args.kernel,args.kernel,args.kernel)) ).astype(img.dtype)

    in_BC = np.where(img!=0)
    out_BC = np.where(out_boundary!=0)

    #Initial Dirichlet boundary conditions

    u[out_BC]= 100
    u[in_BC]= 0

    # Boundary conditions for computing L0 and L1

    L0[:,:,:]= -(dx+dy+dz)/6
    L1[:,:,:]= -(dx+dy+dz)/6

    # Solving the Laplace equation using the Jacobi iterative method

    # determine the region inside which the thickness is to be measured
    R = np.where(out_boundary+img == 0)


    n_iter = 300

    #n_iter = 3*args.radius

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

    #del u

    # Normalization
    np.divide(N_xx, grad_norm, Nx)
    np.divide(N_yy, grad_norm, Ny)
    np.divide(N_zz, grad_norm, Nz)
    # gaussian_filter(Nx, sigma=1, output=Nx)
    # gaussian_filter(Ny, sigma=1, output=Ny)
    # gaussian_filter(Nz, sigma=1, output=Nz)

    del grad_norm, N_xx, N_yy, N_zz
    print("The normalized tangent vector field is successfully computed")
    den = np.absolute(Nx)+ np.absolute(Ny) + np.absolute(Nz)

    den[np.where(den==0)] = 1 ## to avoid dividing by zero

    # iteratively compute correspondence trajectory lengths L0 and L1 using the Gauss_Seidel method

    for it in range (200):
    #for it in range (np.int(0.5*n_iter)):

             L0_n = L0
             L1_n = L1

             L0[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L0_n[(R[0]-np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
             np.absolute(Ny[R[0],R[1],R[2]]) * L0_n[R[0],(R[1]-np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
             + np.absolute(Nz[R[0],R[1],R[2]]) * L0_n[R[0],R[1],(R[2]-np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]

             L1[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L1_n[(R[0]+np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
             np.absolute(Ny[R[0],R[1],R[2]]) * L1_n[R[0],(R[1]+np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
             + np.absolute(Nz[R[0],R[1],R[2]]) * L1_n[R[0],R[1],(R[2]+np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]


             del L0_n, L1_n


    #compute  the thickness of the tissue region inside R

    thickness[R[0],R[1],R[2]] = L1[R[0],R[1],R[2]] + L0[R[0],R[1],R[2]]

    n = nib.Nifti1Image(u, nib.load(args.shape).affine)
    nib.save(n, output+'/harmonic_interpolant.nii.gz')

    p = nib.Nifti1Image(L0, nib.load(args.shape).affine)
    nib.save(p, output+'/L0.nii.gz')

    q = nib.Nifti1Image(L1, nib.load(args.shape).affine)
    nib.save(q, output+'/L1.nii.gz')

    r = nib.Nifti1Image(L0, nib.load(args.shape).affine)
    nib.save(r, output+'/thickness.nii.gz')

    del L0, L1


    #pts = np.zeros(img.shape)

    #pts[points[:,0].astype(int), points[:,1].astype(int), points[:,2].astype(int)] = thickness[points[:,0].astype(int), points[:,1].astype(int), points[:,2].astype(int)]


    texture = thickness[points[:,0].astype(int), points[:,1].astype(int), points[:,2].astype(int)]
    texture/= np.max(texture)
    texture = np.divide(1,1+0.1*np.sqrt(texture))

    del thickness

    #
    # m = nib.Nifti1Image(u, nib.load(args.shape).affine)
    # nib.save(m, './harmonics.nii.gz')

    #feature = np.append(points,texture,axis=1)

    #print(feature.shape)

    feature = np.zeros((points.shape[0],4))
    feature[:,0] = points[:,0]
    feature[:,1] = points[:,1]
    feature[:,2] = points[:,2]
    feature[:,3] = texture#[:]

    #np.save(output+'/'+args.time+'.npy', feature)

    np.save(output+'/feature.npy', feature)


    # res0, res1 = 400, 400
    # mayavi.mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
    # mayavi.mlab.clf()
    # points3d(points[:,0],points[:,1], points[:,2], texture+0.4 , colormap='jet', resolution=60, scale_factor=1)
    # mayavi.mlab.view(azimuth=90, elevation=90, distance=None, focalpoint=None, roll=None, reset_roll=True, figure=None)
    # mayavi.mlab.savefig(output+'/res.png', size=(res0, res1))
    # mayavi.mlab.show()
