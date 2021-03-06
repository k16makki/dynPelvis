import pymesh
import glob
import slam.io as sio

"""
  © Aix Marseille University - LIS-CNRS UMR 7020
  Author(s): Amine Bohi, Karim Makki (amine.bohi,karim.makki{@univ-amu.fr})
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


#--------------------------------------------------------------------
# OBJ to Gifti Converter
# Ref: Makki.K et al. A new geodesic-based feature for characterizationof 3D shapes:
# application to soft tissue organtemporal deformations
# ICPR International Conference on Pattern Recognition -- Milan 01/2021
#
#/----Script Authors------------\
#|                              |
#|       A.Bohi, PhD            |
#|   amine.bohi@univ-amu.fr     |
#|                              |
#\------------------------------/

path = '/home/aminebohi/PycharmProjects/Data/AR_Dyn3D_5SL_QuadMesh_OBJ/'
basename = 'output_AR_Dyn3D_5SL_3dRecbyReg'
mesh_set = glob.glob(path+basename+'*.obj')
mesh_set.sort()

print(len(mesh_set))

for t in range(0,len(mesh_set)):

    print(t)


    prefix = mesh_set[t].split('/')[-1].split('.')[0]

    #print(prefix)

    gifti_file = path+prefix+'.gii'

    print(gifti_file)
    mesh = pymesh.load_mesh(mesh_set[t])
    sio.write_mesh(mesh, gifti_file)








