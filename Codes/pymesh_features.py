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
import pymesh
import glob
import os

data = '/home/karimm/Bureau/marching_cubes/4D_quad_mesh/AR_Dyn3D_5SL_pathological_distorted'

subject = 'AR_Dyn3D_5SL_pathological_distorted'
output = './Dihedral_angle'
#output = './vertex_area'

output = output + '/' + subject

print(output)

if not os.path.exists(output):
   os.makedirs(output)


dynamicSet = glob.glob(data + '/*.obj') 

dynamicSet.sort()
print(dynamicSet)

for  t in range(len(dynamicSet)):
 
        
        prefix = dynamicSet[t].split('/')[-1].split('.')[0]
        print(prefix)
 
        m = pymesh.load_mesh(dynamicSet[t])

        m.add_attribute("vertex_dihedral_angle")
        feature = m.get_attribute("vertex_dihedral_angle")

        #m.add_attribute("vertex_area")
        #feature = m.get_attribute("vertex_area")
        np.save(output+'/'+prefix+'.npy', feature)

      





        

