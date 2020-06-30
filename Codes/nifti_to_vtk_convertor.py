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





# -*- coding: utf-8 -*-

# Import library
#================================================================================================================================================#
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import glob
#================================================================================================================================================#
import logging
logging.basicConfig(level=logging.INFO)

# Useful informations - Global Variables
#================================================================================================================================================#
#------------------------------------------------------------------------------------------------------------------------------------------------#



def distance_to_mask(mask):

    return ndimage.distance_transform_edt(mask)

def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())



def extract_contour(filled_form):

    img = nifti_to_array(filled_form)
    distance_map = distance_to_mask(img)

    contour = np.zeros(img.shape)
    contour[np.where(distance_map==1)]=1

    return contour


# Useful functions
#================================================================================================================================================#
#------------------------------------------------------------------------------------------------------------------------------------------------#
def initTxt(filename):
    try:
        os.remove(filename)
    except:
        pass

#================================================================================================================================================#
#------------------------------------------------------------------------------------------------------------------------------------------------#
def save_contour_as_vtk_file(contour, vtk_out):
    #--------------------------------------------------------------------------------------------------------------------------------------------#
    (sizeX, sizeY, sizeZ) = contour.shape

    #--------------------------------------------------------------------------------------------------------------------------------------------#
    surface_points = np.where(contour)

    nbSurfacePts = surface_points[0].size

    logging.info(surface_points)
    logging.info(nbSurfacePts)

    initTxt(vtk_out)
    vtk_file = open(vtk_out, "a")

    vtk_file.write("# vtk DataFile Version 3.0\n")
    vtk_file.write("vtk output\n")
    vtk_file.write("ASCII\n")
    vtk_file.write("DATASET POLYDATA\n")

    vtk_file.write("POINTS {} float\n".format(nbSurfacePts))

    #--------------------------------------------------------------------------------------------------------------------------------------------#

    for c in range(nbSurfacePts):
        vtk_file.write("{:.1f} {:.1f} {:.1f}\n".format(surface_points[0][c], surface_points[1][c], surface_points[2][c]))

    vtk_file.write("VERTICES {} {}\n".format(nbSurfacePts, nbSurfacePts*2))

    for c in range(nbSurfacePts):
        vtk_file.write("1 {}\n".format(c))

    #--------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#




PATH_GLOBAL = "/home/karimm/Bureau/Deep_project/AR_patho"


sequence_name= 'simulated_distorted_sequence'

sequence_path = PATH_GLOBAL + '/' + sequence_name

output_path = sequence_path+'_vtk'
if not os.path.exists(output_path):
       os.makedirs(output_path)

nifti_dynamicSet = glob.glob(sequence_path+'/'+'*.nii.gz')
nifti_dynamicSet.sort()

print(nifti_dynamicSet)


# Main
#================================================================================================================================================

for i in range(len(nifti_dynamicSet)):
    prefix = nifti_dynamicSet[i].split('/')[-1].split('.')[0]
    vtk_out = output_path+'/'+prefix+'.vtk'

    contour = extract_contour(nifti_dynamicSet[i])

    save_contour_as_vtk_file(contour, vtk_out)


    #convertFormat_3D_nii_to_vtk(nifti_dynamicSet[i], outfile)

# inn = '/home/karim/Bureau/regular_geodesics.nii.gz'
# out = '/home/karim/Bureau/mayavi2_example.vtk'
#
# convertFormat_3D_nii_to_vtk(inn, out)
