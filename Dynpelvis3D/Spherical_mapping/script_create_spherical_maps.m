clc, clear all, close all

"""
  © Aix Marseille University - LIS-CNRS UMR 7020
  Author(s): Amine Bohi, Karim Makki (amine.bohi, karim.makki{@univ-amu.fr})
  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  http://www.cecill.info.
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


%--------------------------------------------------------------------
% Matlab script for computing and saving spherical maps
% of several meshes corresponding to a subject dynamique sequence 
% 


addpath gifti\
%addpath('External_packages\')

genpath = 'C:\Users\bohi.a\PycharmProjects\Data\';

dir_to_mesh = [genpath 'MESH_DATA\trimesh_gifti_all\AR_Dyn3D_5SL\'];% mesh folder
dir_to_mean_curv = [genpath 'Features\mean_curvature_all\AR_Dyn3D_5SL\']; % mean curvature folder
dir_to_gauss_curv = [genpath 'Features\gauss_curvature_all\AR_Dyn3D_5SL\']; % gaussian curvature folder
dir_to_distortion = [genpath 'Features\distortion_all\AR_Dyn3D_5SL\']; % distortion folder
dir_to_elongation = [genpath 'Features\elongation_all\AR_Dyn3D_5SL\']; % elongation folder

txtpattern1 = fullfile(dir_to_mesh, '*.gii');
d1=dir(txtpattern1); % capture everything in the directory with gii extension
allNames1={d1.name}; % extract names of all gii-files
allNames1=natsortfiles(allNames1);

txtpattern2 = fullfile(dir_to_mean_curv, '*.npy');
d2=dir(txtpattern2); % capture everything in the directory with npy extension
allNames2={d2.name}; % extract names of all npy-files
allNames2=natsortfiles(allNames2);

txtpattern3 = fullfile(dir_to_gauss_curv, '*.npy');
d3=dir(txtpattern3); % capture everything in the directory with npy extension
allNames3={d3.name}; % extract names of all npy-files
allNames3=natsortfiles(allNames3);

txtpattern4 = fullfile(dir_to_distortion, '*.npy');
d4=dir(txtpattern4); % capture everything in the directory with npy extension
allNames4={d4.name}; % extract names of all npy-files
allNames4=natsortfiles(allNames4);

txtpattern5 = fullfile(dir_to_elongation, '*.npy');
d5=dir(txtpattern5); % capture everything in the directory with npy extension
allNames5={d5.name}; % extract names of all npy-files
allNames5=natsortfiles(allNames5);



for i=1:length(allNames1)% for each surface

    FV = load_mesh(fullfile(dir_to_mesh, allNames1{i}));
    
    mean_curv = readNPY(fullfile(dir_to_mean_curv, allNames2{i}));
    fullfile(dir_to_mean_curv, allNames2{i})
    
    gauss_curv = readNPY(fullfile(dir_to_gauss_curv, allNames3{i}));
    fullfile(dir_to_gauss_curv, allNames3{i})
    
    distortion = readNPY(fullfile(dir_to_distortion, allNames4{i}));
    fullfile(dir_to_distortion, allNames4{i})
    
    elongation = readNPY(fullfile(dir_to_elongation, allNames5{i}));
    fullfile(dir_to_elongation, allNames5{i})
    
    % compute nodal connectivity
    VertConn=vertices_connectivity(FV);

    % map to sphere (the adapted appraoch published in EMBC 2019)
    [sphFV,V]=map_sphere(FV,[1 2 3],VertConn);

    base=strtok(allNames1{i},'.'); % chop off the extension (".gii")
    filename = strcat(base, '.mat');
    mkdir('C:\Users\bohi.a\PycharmProjects\Data\spherical_maps\TV_Dyn3D_5SL\\');%Save ALL properties
    save(['C:\Users\bohi.a\PycharmProjects\Data\spherical_maps\TV_Dyn3D_5SL\\' filename], 'sphFV','FV','VertConn','mean_curv','gauss_curv','distortion','elongation');
         
end