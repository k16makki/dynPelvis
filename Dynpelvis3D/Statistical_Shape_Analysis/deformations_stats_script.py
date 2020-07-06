import numpy as np
import matplotlib as mpl
import glob
import matplotlib.pyplot as plt


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
# script for calculating the correlations or other metrics for an intra-subject comparison

# Ref: Makki.K et al. A new geodesic-based feature for characterizationof 3D shapes:
# application to soft tissue organtemporal deformations
# ICPR International Conference on Pattern Recognition -- Milan 01/2021
#



output_path = '../Data/Features/Stats/temporal_corr/gdist_corr/'
basename = 'output_AR_Dyn3D_5SL_3dRecbyReg'
feature_set = glob.glob('../Data/Features/geodesic_feature_all/AR_Distorted/'+basename+'*.npy')
feature_set.sort()

correlation = np.zeros(len(feature_set)-1)
print(len(feature_set))

data_0 = np.load(feature_set[0])

for t in range(len(feature_set)-1):


    data_i = np.load(feature_set[t+1])

    # euclidean distance
    #diff = np.absolute(data_i - data_0)

    # correlation
    corr = np.corrcoef(data_0[...,3], data_i[...,3])

    #mean_abs_err[t] = np.divide(np.sum(np.abs(data_i - data_0)), len(data_i))
    #mean_rel_err[t] = np.divide(np.sum(np.abs(data - data_def)), 2 *
    #                            (np.maximum(np.sum(np.abs(data)), np.sum(np.abs(data_def)))))
    #rpd_err[t] = np.divide(np.sum(np.divide(np.abs((data - data_def)),
    #                                 (np.abs(data + np.abs(data_def))))), len(data))

    correlation[t] = corr[0, 1]
    #diff_mat[t] = np.mean(diff)

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))
# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
#bp = ax.boxplot(diff_mat[:,np.arange(0,200,10)], showfliers=False)
#plt.xticks(np.arange(0, len(feature_set) , 50))

plt.plot(correlation, 'b--', linewidth=1, markersize=5)
plt.xlabel('Time Frames')
plt.ylabel('Correlations')
plt.title('AR Distorted')
plt.show()