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

opath = '/home/karimm/Bureau/Deep_project/AR_patho/simulated_distorted_sequence/'


basename= 'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'
dynamicSet = glob.glob(opath+basename+'*.nii.gz')
dynamicSet.sort()

#print(dynamicSet)

cycle_length = len(dynamicSet) 

#print(cycle_length)



for t in range (cycle_length):
	#print(dynamicSet[t])
	a = "{:04d}".format(t+cycle_length)
	print(a)

	new_image = opath+'AR_Dyn3D_5SL_3dRecbyReg-Expi_FilledContour_'+"{:04d}".format(t+cycle_length)+'.nii.gz'

	go = 'cp '+  dynamicSet[t] + ' ' + new_image
	print(go)
	os.system(go)


