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
# script for computing area distortions
#



path = '../Data/AF_Dyn3D_5SL_QuadMesh_OBJ/'
basename = 'output_AF_Dyn3D_5SL_3dRecbyReg'
mesh_set = glob.glob(path + basename + '*.obj')
mesh_set.sort()
print(mesh_set)

mesh_0 = pymesh.load_mesh("AF_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_0000.obj")
mesh_0.add_attribute("face_area")
areas = mesh_0.get_attribute("face_area")
print(areas)

areas_ratio_0 = np.divide(areas, sum(areas))
print(areas_ratio_0)


for t in range(len(mesh_set)):
    mesh_i = pymesh.load_mesh(mesh_set[t])
    mesh_i.add_attribute("face_area")
    areas_i = mesh_i.get_attribute("face_area")
    areas_ratio_i = np.divide(areas_i, sum(areas_i))

    areas_dist = areas_ratio_0 - areas_ratio_i


    resulting_file = '../Data/mesh_areadist_eval' + '/' + 'AF_3Dyn_5SL_'+np.str(t) + '.npy'
    np.save(resulting_file, areas_dist)

