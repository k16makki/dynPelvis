import pymesh
import numpy as np
import glob

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
# script for calculating the 3D elongations
# based on the differences of the averages of the areas of the 4 faces
# adjacent to the vertex of interest.
# Ref: Makki.K et al. A new geodesic-based feature for characterizationof 3D shapes:
# application to soft tissue organtemporal deformations
# ICPR International Conference on Pattern Recognition -- Milan 01/2021
#



output_path = '../Data/Features/elongation_all/'
path = '../Data/AF_Dyn3D_5SL_QuadMesh_OBJ/'
basename = 'output_AF_Dyn3D_5SL_3dRecbyReg'
mesh_set = glob.glob(path + basename + '*.obj')
mesh_set.sort()
print(mesh_set)

mesh_0 = pymesh.load_mesh("AF_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_0000.obj")
mesh_0.add_attribute("vertex_area")
vert_areas_0 = mesh_0.get_attribute("vertex_area")
print(vert_areas_0.shape)


for t in range(len(mesh_set)-1):
    #mesh_i = pymesh.load_mesh(mesh_set[t])
    mesh_j = pymesh.load_mesh(mesh_set[t])

    mesh_j.add_attribute("vertex_area")
    vert_areas_j = mesh_j.get_attribute("vertex_area")

    #mesh_j.add_attribute("vertex_area")
    #vert_areas_j = mesh_j.get_attribute("vertex_area")

    elongation = vert_areas_j - vert_areas_0

    prefix = mesh_set[t].split('/')[-1].split('.')[0]
    print(prefix)

    resulting_file = output_path + prefix + '.npy'
    np.save(resulting_file, elongation)

