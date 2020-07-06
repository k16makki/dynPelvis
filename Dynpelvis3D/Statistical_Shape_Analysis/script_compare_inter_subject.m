clc, clear all, close all

%addpath('External_packages\')

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


%--------------------------------------------------------------------
% Matlab script for computing distances matrices based on the differences 
% of maximal temporal deformations between each pair of 2 subjects
% 
% Max temporal deformation == Min of correlation along the dynamique sequence
%


genpath = 'C:\Users\bohi.a\PycharmProjects\Data\';
output_path = [genpath 'Results\Statistics\inter_subjects_stats\'];
dir_to_search = [genpath 'Features\Stats\temporal_corr\gdist_corr\']; % tex correlation folder 

txtpattern = fullfile(dir_to_search, '*.npy');
d=dir(txtpattern); % capture everything in the directory with npy extension
allNames={d.name}; % extract names of all npy-files
allNames=natsortfiles(allNames); % sort files by name

for i=1:length(allNames)% for each surface
    
    out_express1 = strsplit(allNames{i},'.');
    out_express2 = strsplit(out_express1{1},'_');
    subject_name = out_express2{1};% To get the name of the current subject

    tex = readNPY(fullfile(dir_to_search, allNames{i}));% Feature vector
    allNames{i}
    
    if isequal(subject_name,'AF')
        feature.af = min(tex);% Max temporal deformation of AF
    elseif isequal(subject_name,'AO')
        feature.ao = min(tex);% AO
    elseif isequal(subject_name,'AOL')
        feature.aol = min(tex);% AOL
    elseif isequal(subject_name,'AR')
        feature.ar = min(tex);% AR
    elseif isequal(subject_name,'ARP')
        feature.arp = min(tex);% AR simulated
    elseif isequal(subject_name,'CM')
        feature.cm = min(tex);% CM
    elseif isequal(subject_name,'NF')
        feature.nf = min(tex);% NF
    elseif isequal(subject_name,'TV')
        feature.tv = min(tex);%TV
    end 
        
end

% L2 distances between subjects
af_ao_dist = L2_distance(feature.af,feature.ao); af_aol_dist = L2_distance(feature.af,feature.aol);
af_ar_dist = L2_distance(feature.af,feature.ar); af_arp_dist = L2_distance(feature.af,feature.arp);
af_cm_dist = L2_distance(feature.af,feature.cm); af_nf_dist = L2_distance(feature.af,feature.nf);
af_tv_dist = L2_distance(feature.af,feature.tv); 
ao_aol_dist = L2_distance(feature.ao,feature.aol); ao_ar_dist = L2_distance(feature.ao,feature.ar);
ao_arp_dist = L2_distance(feature.ao,feature.arp); ao_cm_dist = L2_distance(feature.ao,feature.cm);
ao_nf_dist = L2_distance(feature.ao,feature.nf); ao_tv_dist = L2_distance(feature.ao,feature.tv);
aol_ar_dist = L2_distance(feature.aol,feature.ar); aol_arp_dist = L2_distance(feature.aol,feature.arp);
aol_cm_dist = L2_distance(feature.aol,feature.cm); aol_nf_dist = L2_distance(feature.aol,feature.nf); 
aol_tv_dist = L2_distance(feature.aol,feature.tv);
ar_arp_dist = L2_distance(feature.ar,feature.arp); ar_cm_dist = L2_distance(feature.ar,feature.cm); 
ar_nf_dist = L2_distance(feature.ar,feature.nf); ar_tv_dist = L2_distance(feature.ar,feature.tv);
arp_cm_dist = L2_distance(feature.arp,feature.cm); arp_nf_dist = L2_distance(feature.arp,feature.nf); 
arp_tv_dist = L2_distance(feature.arp,feature.tv);
cm_nf_dist = L2_distance(feature.cm,feature.nf); cm_tv_dist = L2_distance(feature.cm,feature.tv);
nf_tv_dist = L2_distance(feature.nf,feature.tv);


D_dist = [af_ao_dist af_aol_dist af_ar_dist af_arp_dist af_cm_dist af_nf_dist af_tv_dist ...
          ao_aol_dist ao_ar_dist ao_arp_dist ao_cm_dist ao_nf_dist ao_tv_dist ...
          aol_ar_dist aol_arp_dist aol_cm_dist aol_nf_dist aol_tv_dist ...
          ar_arp_dist ar_cm_dist ar_nf_dist ar_tv_dist ...
          arp_cm_dist arp_nf_dist arp_tv_dist ...
          cm_nf_dist cm_tv_dist ...
          nf_tv_dist];
    
dist_mat = squareform(D_dist);
dist_mat = dist_mat/max(max(dist_mat));%we normalize the distance matrix
%dist_mat(dist_mat==0)=1

% options.dims = 1:6; %(row) vector of embedding dimensionalities to use (1:10 = default)
% options.comp = 1; %which connected component to embed, if more than one. (1 = largest (default), 2 = second largest, ...)
% options.display = 1; %plot residual variance and 2-D embedding? (1 = yes (default), 0 = no)
% options.overlay = 1; %overlay graph on 2-D embedding? (1 = yes (default), 0 = no)
% options.verbose = 1; %display progress reports? (1 = yes (default), 0 = no)
% % options.label = ['rus', 'pey', 'tau', 'pet', 'don', 'mai']
% 
% Y_corr = Isomap(dist_mat, 'k', 5, options); 

%Plot the distance matrix
names = {'AF'; 'AO'; 'AOL'; 'AR'; 'AR\_P'; 'CM'; 'NF'; 'TV'};

surf(dist_mat, 'FaceColor','texturemap',...
   'EdgeColor','none',...
   'FaceLighting','none')
% pcolor(dist_mat)
set(gca,'xtick',[1:8],'xticklabel',names)
set(gca,'ytick',[1:8],'yticklabel',names)
view(0,90)
title('Inter-Subject Comparison -- Geodesic feature')
colormap jet
colorbar
