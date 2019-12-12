
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
import trimesh
import os
import math
#from numpy.linalg import det
from numpy.linalg import inv
from scipy import ndimage
import gdist
from visbrain.objects import BrainObj, ColorbarObj, SceneObj
import argparse
from scipy.spatial.distance import cdist




def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())



def load_mesh(gii_file):
    """
    load gifti_file and create a trimesh object
    :param gifti_file: str, path to the gifti file
    :return: the corresponding trimesh object
    """
    g = nib.gifti.read(gii_file)
    vertices, faces = g.getArraysFromIntent(
        nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data, \
        g.getArraysFromIntent(
            nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
    metadata = g.get_meta().metadata
    metadata['filename'] = gii_file

    return trimesh.Trimesh(faces=faces, vertices=vertices,
                           metadata=metadata, process=False)


def mesh_to_stl(save_path, vertices, faces):

  """
  Save surface mesh structure into an stl mesh.
  save_path          : Path of mesh
  vertices           : Coordinates of all surface nodes
  f_indices          : Indices of nodes of all surface triangle faces
  """

  # Create the .stl mesh par Trimesh and save it
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
  mesh.export(save_path)


def texture_plot(mesh, tex=None, caption=None, cblabel=None, visb_sc=None,
                  cmap='gnuplot'):
    """
    Projecting Texture onto trimesh object using visbrain core plotting tool
    :param mesh: trimesh object
    :param tex: numpy array of a texture to be visualized
    :return: 0
    """

    b_obj = BrainObj('gui', vertices=np.array(mesh.vertices),
                     faces=np.array(mesh.faces),
                     translucent=False)
    if visb_sc is None:
        visb_sc = SceneObj(bgcolor='black', size=(1400, 1000))
        visb_sc.add_to_subplot(b_obj, title=caption)
        visb_sc_shape = (1, 1)
    else:
        visb_sc_shape = get_visb_sc_shape(visb_sc)
        visb_sc.add_to_subplot(b_obj, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1], title=caption)

    if tex is not None:
        b_obj.add_activation(data=tex, cmap=cmap,
                             clim=(np.min(tex), np.max(tex)))
        CBAR_STATE = dict(cbtxtsz=20, txtsz=20., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), cblabel=cblabel)
        cbar = ColorbarObj(b_obj, **CBAR_STATE)
        visb_sc.add_to_subplot(cbar, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1] + 1, width_max=200)
    return visb_sc



def distance_to_mask(mask):

    return ndimage.distance_transform_edt(mask)


def extract_contour(filled_form):

    distance_map = distance_to_mask(filled_form)
    contours = np.zeros(distance_map.shape)
    contours[np.where(distance_map==1)]=1

    return contours




def determine_shape_poles(mask):

    """
    Determine organ poles using PCA
    :param mask: numpy array of organ binary mask
    :return: North pole, south pole, east pole, and west pole.
    North and south poles correspond to the intersection points
    between the second axis of inertia and the 3D shape contours, while
    the east and west poles correspond to the intersection points
    between the principal axis of inertia and the 3D shape contours
    """

    nii = nib.load(mask)
    shape = nii.get_data()

    x, y, z = np.nonzero(shape)
    mesh_centroid=[int(np.mean(x)), int(np.mean(y)),int(np.mean(z))]

    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    coords = np.vstack([x, y, z])

    # Covariance matrix and its eigenvectors and eigenvalues

    cov = np.cov(coords)
    eig_vals, eig_vecs = np.linalg.eig(cov)

    sort_indices = np.argsort(eig_vals)[::-1]

    x_v1, y_v1, z_v1 = eig_vecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2, z_v2 = eig_vecs[:, sort_indices[1]]
    x_v3, y_v3, z_v3 = eig_vecs[:, sort_indices[2]]

    principal_axis_of_inertia = np.zeros(shape.shape)
    second_axis_of_inertia = np.zeros(shape.shape)

    line_length = 25
    ### Tracer la droite passante par le centroide et de vecteur directeur l'axe principal d'inertie

    i = np.linspace(0,2*line_length-1,2*line_length)
    principal_axis_of_inertia[(mesh_centroid[0]-i*x_v1).astype(int), (mesh_centroid[1]-i*y_v1).astype(int), (mesh_centroid[2]-i*z_v1).astype(int)]=1
    principal_axis_of_inertia[(mesh_centroid[0]+i*x_v1).astype(int), (mesh_centroid[1]+i*y_v1).astype(int), (mesh_centroid[2]+i*z_v1).astype(int)]=1

    second_axis_of_inertia[(mesh_centroid[0]-i*x_v2).astype(int), (mesh_centroid[1]-i*y_v2).astype(int), (mesh_centroid[2]-i*z_v2).astype(int)]=1
    second_axis_of_inertia[(mesh_centroid[0]+i*x_v2).astype(int), (mesh_centroid[1]+i*y_v2).astype(int), (mesh_centroid[2]+i*z_v2).astype(int)]=1

    # Binary dilation to ensure line continuity

    principal_axis_of_inertia = ndimage.binary_dilation(principal_axis_of_inertia).astype(principal_axis_of_inertia.dtype)
    second_axis_of_inertia = ndimage.binary_dilation(second_axis_of_inertia).astype(second_axis_of_inertia.dtype)

    # Detection of intersection points

    ## Intersection pointset: between the principal axis of inertia and the organ 3D contours
    shape_contour = extract_contour(shape)

    intersection_set = np.argwhere(np.logical_and(principal_axis_of_inertia!=0, shape_contour!=0))

    ## Matrice de distances entre les points d'intersections, deux a deux

    distance_matrix = cdist(intersection_set, intersection_set, 'euclidean')

    ## poles estet ouest

    east_pole =  intersection_set[0]
    west_pole = intersection_set[np.argmax(distance_matrix[0,:])]

    print("east pole:\n")

    print(east_pole)

    print("west pole:\n")

    print(west_pole)

    ## Liste des points d'intersection entre le second axe principal d'inertie et la surface (contour) 3D de l'organe

    intersection_set1 = np.argwhere(np.logical_and(second_axis_of_inertia!=0, shape_contour!=0))

    ## Matrice de distances Euclidiennes entre les points d'intersections, deux a deux

    distance_matrix1 = cdist(intersection_set1, intersection_set1, 'euclidean')

    ## pole est

    north_pole =  intersection_set1[0]
    south_pole = intersection_set1[np.argmax(distance_matrix1[0,:])]

    print("north pole:\n")

    print(north_pole)

    print("south pole:\n")

    print(south_pole)


    return(north_pole, south_pole, east_pole, west_pole)



# Conversion from stl to gifti :
# Example using freesurfer: mris_convert  /home/karim/Bureau/sphere_example.stl /home/karim/Bureau/sphere_example.gii

# Convert  texture from mesh .gifti to image .nii.gz
def texture_gifti_to_nifti(outpath, gifti_name, filename_nii_reso, texture1, texture2, niiname=None):

  """
  Save mesh of type gifti to a nifti file including a binary map.
  The geodesic distance maps will be saved as nifti files.
  outpath            : Output directory
  gifti_name         : Input mesh name in .gii format
  filename_nii_reso  : Reference nifti of output nifti binary map
  niiname            : Output nifti file
  texture1           : Longitudinal geodesics map
  texture2           : Transverse geodesics map
  """
  ref_header = nib.load(filename_nii_reso).header

  # load gifti mesh

  mesh = load_mesh(gifti_name)

    # Load mesh vertices

  points = mesh.vertices

  # Convert to binary image
  arr_reso = nib.load(filename_nii_reso).get_data()
  geodesics0 = np.zeros(arr_reso.shape)
  geodesics1 = np.zeros(arr_reso.shape)

# Compute coordinates in the reference space
  coords_ref = np.zeros((len(points[...,0]),3),dtype='float64')

  print("vertex_id:\n")
  print(coords_ref[0,:].astype(int))
# Apply the affine transformation of the reference image: goes back from world coordinates to image coordinates
  aff = inv(nib.load(filename_nii_reso).affine)
  coords_ref[:,0] = aff[0,0]*points[...,0] + aff[0,1]*points[...,1] + aff[0,2]* points[...,2] + aff[0,3]
  coords_ref[:,1] = aff[1,0]*points[...,0] + aff[1,1]*points[...,1] + aff[1,2]* points[...,2] + aff[1,3]
  coords_ref[:,2] = aff[2,0]*points[...,0] + aff[2,1]*points[...,1] + aff[2,2]* points[...,2] + aff[2,3]

# writing geodesic textures as numpy arrays with the same dimensions of that of the reference nifti image

  geodesics0[(coords_ref[:,0]).astype(int), (coords_ref[:,1]).astype(int), (coords_ref[:,2]).astype(int)] = texture1
  geodesics1[(coords_ref[:,0]).astype(int), (coords_ref[:,1]).astype(int), (coords_ref[:,2]).astype(int)] = texture2

# Compute the downsampled regular mesh

  epsilon0 = 0.8
  epsilon = 1  #2
  downsampled_mesh = np.zeros(arr_reso.shape)

  ## percentage of downsampling: this will keep 25% of the available information or of the full mesh

  percentage = 20

  nb_long_geodesics = int((np.max(texture1)*percentage)/100) # number of longitudinal geodesics
  nb_trans_geodesics = int((np.max(texture2)*percentage)/100) # number of transversal geodesics

  lines1 = np.linspace(1, np.max(texture1), nb_long_geodesics)
  lines2 = np.linspace(1, np.max(texture2), nb_trans_geodesics)


  for i in np.nditer(lines1):

      downsampled_mesh[np.where(np.logical_and(geodesics0 <= i+epsilon0, geodesics0 >= i-epsilon0))] = 1

  for i in np.nditer(lines2):

      downsampled_mesh[np.where(np.logical_and(geodesics1 <= i+epsilon, geodesics1 >= i-epsilon))] += 1


  downsampled_mesh[np.where(downsampled_mesh<2)] = 0

  ## Save outputs as nifti files

  i = nib.Nifti1Image(downsampled_mesh, aff)
  j = nib.Nifti1Image(geodesics0, aff)
  k = nib.Nifti1Image(geodesics1, aff)

  nib.save(j, os.path.join(outpath, 'longitudinal_geodesics.nii.gz'))
  nib.save(k, os.path.join(outpath, 'transverse_geodesics.nii.gz'))

  if niiname is not None:
      nib.save(i, niiname)
  else:
      nib.save(i, os.path.join(outpath, 'downsampled_mesh.nii.gz'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='dynPelvis')
    parser.add_argument('-in', '--input', help='input gifti file', type=str, required = True)
    parser.add_argument('-ref', '--reference', help='input nifti file', type=str, required = True)
    parser.add_argument('-opath', '--output', help='output directory', type=str, required = True)

    args = parser.parse_args()

    print("ok")

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    #determine_shape_poles(args.reference)

    mesh = load_mesh(args.input)

    print(mesh.metadata)

    mesh.apply_transform(mesh.principal_inertia_transform)

    vert_id = 0
    vert_id1 = 0

    print("Centroid:\n")
    print(mesh.centroid)

    print("Principal inertia vectors:\n")
    print(mesh.principal_inertia_vectors)

    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)


    # Compute longitudinal geodesics on the organ surfaces: compute the geodesic distance from one point to all vertices on mesh

    source_index = np.array([vert_id], dtype=np.int32)


    target_index = np.linspace(0, len(vert)-1, len(vert)).astype(np.int32)
    long_geodesics = gdist.compute_gdist(vert, poly, source_index, target_index)

    # Compute transverse geodesics on the organ surfaces
    source_index1 = np.array([vert_id1], dtype=np.int32)
    trans_geodesics = gdist.compute_gdist(vert, poly, source_index1, target_index)


    texture_gifti_to_nifti(args.output,args.input, args.reference, long_geodesics, trans_geodesics, niiname=None)

    print("Mesh vertices shape:\n")
    print(mesh.vertices.shape)



    visb_sc = texture_plot(mesh=mesh, tex=trans_geodesics,
                                 caption='Transverse geodesics',
                                 cblabel='Transverse_geodesics')

    visb_sc.preview()
