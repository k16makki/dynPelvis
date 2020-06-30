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



vtk_path = '/home/karimm/Bureau/input_data/output/AR_patho_distorted'# global path to Deformetrica's output
nifti_path = '/home/karimm/Bureau/Deep_project/AR_patho/simulated_distorted_sequence'
nifti_basename = 'AR_Dyn3D_5SL_3dRecbyReg'#-Expi'#_FilledContour'
#vtk_basename = 'AOL_Dyn3D_5SL_3dRecbyReg-Expi'#_Contour'


nifti_dynamicSet = glob.glob(nifti_path+'/'+nifti_basename+'*.nii.gz')
nifti_dynamicSet.sort()

print(nifti_dynamicSet)



vtk_basename = 'output_' + nifti_basename

vtk_dynamicSet = glob.glob(vtk_path+'/'+vtk_basename+'*')
vtk_dynamicSet.sort()

print(vtk_dynamicSet)

for t in range(len(vtk_dynamicSet)-1):

    prefix = vtk_dynamicSet[t].split('/')[-1].split('.')[0]


    points = vtk_dynamicSet[t]+'/DeterministicAtlas__Reconstruction__bladder__subject_subj1.vtk'

    volume = nifti_dynamicSet[t+1]

    go = 'time python sphere_projection_simulations.py -in ' + volume + ' -subject ' + nifti_basename+'_pathological_distorted' + ' -t ' + prefix + ' -pts ' + points

    print(go)

    os.system(go)


    #print(points)
