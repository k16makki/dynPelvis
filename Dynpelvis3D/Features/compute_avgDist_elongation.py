import numpy as np
import glob
from sklearn.neighbors import NearestNeighbors
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
# script for calculating the 3D elongations
# based on the differences of the averages of the distances
# between the vertex of interest and its 4 neighbors since we are working on
# quadrilateral configurations
# Ref: Makki.K et al. A new geodesic-based feature for characterizationof 3D shapes:
# application to soft tissue organtemporal deformations
# ICPR International Conference on Pattern Recognition -- Milan 01/2021



output_path = '../Data/Features/elongation_all/'
path = '../Data/AF_Dyn3D_5SL_QuadMesh_Gifti/'
basename = 'output_AF_Dyn3D_5SL_3dRecbyReg'
mesh_set = glob.glob(path + basename + '*.gii')
mesh_set.sort()
print(mesh_set)

data_ref = sio.load_mesh('QuadMesh_AF.gii')
pcd_pts_ref = data_ref.vertices

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(pcd_pts_ref)

distances, indices = nbrs.kneighbors(pcd_pts_ref)
print(distances.shape)
print(indices.shape)
average_dist_ref = np.mean(distances[:, 1:5], axis=1)
print(average_dist_ref)
distances_trg_i = np.zeros(distances.shape)
distances_trg_j = np.zeros(distances.shape)

for t in range(len(mesh_set)-1):
    data_dist_i = sio.load_mesh(mesh_set[t])
    pcd_pts_dist_i = data_dist_i.vertices
    data_dist_j = sio.load_mesh(mesh_set[t+1])
    pcd_pts_dist_j = data_dist_j.vertices

    distances_trg_i[:, 0] = 0
    distances_trg_i[..., 1] = np.linalg.norm(pcd_pts_dist_i[indices[..., 0]] - pcd_pts_dist_i[indices[..., 1]], axis=1)
    distances_trg_i[..., 2] = np.linalg.norm(pcd_pts_dist_i[indices[..., 0]] - pcd_pts_dist_i[indices[..., 2]], axis=1)
    distances_trg_i[..., 3] = np.linalg.norm(pcd_pts_dist_i[indices[..., 0]] - pcd_pts_dist_i[indices[..., 3]], axis=1)
    distances_trg_i[..., 4] = np.linalg.norm(pcd_pts_dist_i[indices[..., 0]] - pcd_pts_dist_i[indices[..., 4]], axis=1)

    distances_trg_j[:, 0] = 0
    distances_trg_j[..., 1] = np.linalg.norm(pcd_pts_dist_j[indices[..., 0]] - pcd_pts_dist_j[indices[..., 1]], axis=1)
    distances_trg_j[..., 2] = np.linalg.norm(pcd_pts_dist_j[indices[..., 0]] - pcd_pts_dist_j[indices[..., 2]], axis=1)
    distances_trg_j[..., 3] = np.linalg.norm(pcd_pts_dist_j[indices[..., 0]] - pcd_pts_dist_j[indices[..., 3]], axis=1)
    distances_trg_j[..., 4] = np.linalg.norm(pcd_pts_dist_j[indices[..., 0]] - pcd_pts_dist_j[indices[..., 4]], axis=1)

    # average_dist_trg_i = np.mean(distances_trg_i[:, 1:5], axis=1)
    # average_dist_trg_j = np.mean(distances_trg_j[:, 1:5], axis=1)
    average_dist_trg_i = distances_trg_i[:, 1]
    average_dist_trg_j = distances_trg_j[:, 1]
    elongation = average_dist_trg_j - average_dist_trg_i

    prefix = mesh_set[t].split('/')[-1].split('.')[0]
    print(prefix)

    resulting_file = output_path + prefix + '.npy'

    np.save(resulting_file, elongation)

print(np.min(elongation), np.max(elongation))

# x = pcd_pts_ref[:, 0]
# y = pcd_pts_ref[:, 1]
# z = pcd_pts_ref[:, 2]
#
# mayavi.mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
# mayavi.mlab.clf()
# points3d(x, y, z, elongation, colormap='jet', resolution=300, scale_factor=1)
# # time.sleep(2)
# #mayavi.mlab.savefig('diff_curvatures4.png', size=(400, 400))
# # pts.glyph.glyph.clamping = True
# mayavi.mlab.view(azimuth=90, elevation=90, distance=None, focalpoint=None, roll=None, reset_roll=True, figure=None)
# lut_manager = mayavi.mlab.colorbar(orientation='vertical')
# mayavi.mlab.show()
