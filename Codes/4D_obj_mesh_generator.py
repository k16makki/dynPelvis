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


import pymesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob
import os

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

# Mesh source (quad mesh generated using Instant Meshes, .obj format) 

mesh_src = pymesh.load_mesh("/home/karimm/Bureau/marching_cubes/AR_Dyn3D_patho_simulated/AR_patho_new_M0.obj")

# Path to the folder containing the output of Deormetrica

path = '/home/karimm/Bureau/input_data/output/AR_patho_distorted/'

# Basename from the subfolders of the Deformetrica main Output folder
basename = 'output_AR_Dyn3D_5SL_3dRecbyReg'

#subject_name
Subject_name = 'AR_Dyn3D_5SL_pathological_distorted'


points = mesh_src.vertices

print(points.shape)
#result_path = output_path+'/'+basename+'_elongation'
vertices_set = glob.glob(path+basename+'*')
vertices_set.sort()

print(len(vertices_set))

outfile =  './'+Subject_name
if not os.path.exists(outfile):
   os.makedirs(outfile)

for t in range(0,len(vertices_set)-1):

    #print(t)

    prefix = vertices_set[t].split('/')[-1].split('.')[0]

    tracked_pointset_vtk = vertices_set[t]+'/DeterministicAtlas__Reconstruction__bladder__subject_subj1.vtk'

    #print(tracked_pointset_vtk)

   
    points = vtk_to_array(tracked_pointset_vtk)
    
    mesh_trg = pymesh.form_mesh(points, mesh_src.faces)

    pymesh.save_mesh(outfile+'/'+prefix+'.obj', mesh_trg)

#print(points.shape)
