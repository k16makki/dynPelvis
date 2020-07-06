import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
import glob

"""
  © Aix Marseille University - LIS-CNRS UMR 7020
  Author(s): Karim Makki, Amine Bohi (karim.makki, amine.bohi{@univ-amu.fr})
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


# The current folder should contain all the volumes V0 (as nifti files) and the current script must be in the same folder containing the nifti volume(s)
# This will output a set of mesh surfaces of the organ


subjectSet = glob.glob('./*.nii.gz') 

subjectSet.sort()
print(subjectSet)

for subject in range(len(subjectSet)):
 
        vol = nib.load(subjectSet[subject]).get_data()
        subject_name = subjectSet[subject].split('/')[-1].split('.')[0]
 
        outfile_name = './'+subject_name+'.ply'

        verts, faces, norm, val = measure.marching_cubes_lewiner(vol, spacing=(1,1,1), gradient_direction='ascent')#,allow_degenerate=True)

        mesh = trimesh.Trimesh(vertices= verts, faces= faces) 

        mesh.export(outfile_name)

        

