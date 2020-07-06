function [sphFV,V,nd_c,VertConn,D]=map_sphere(FV,indices,VertConn,V)
%--------------------------------------------------------------------
% Adaptative Spherical mapping with the first eigenfunctions associated 
% to a larger eigenvalues set in the spectrum
% with only 2 nodal domains
% Ref: Bohi A. et al. Global Perturbation of Initial Geometry in a Biomechanical Model of
% Cortical Morphogenesis
% 41st EMBC 2019 -- Berlin
%
%/---------------------Script Authors----------------------------\
%|                                                               | 
%|       A.Bohi, PhD          &&      J.Lefèvre                  |  
%|   amine.bohi@univ-amu.fr   && julien.lefevre@univ-amu.fr      |
%|                                                               | 
%\---------------------------------------------------------------/

% 6 first LBO eigenfunctions computation
if nargin<4
    [A2,G,grad_v,aires,index1,index2,cont_vv]=heat_matrices(FV.faces,FV.vertices,3,1);
    [V,D]=eigs(A2,G,6,'sm');
end

V_nd = fliplr(V);

if nargin==3
    if size(VertConn)==1
        nd_c=[0 0 0];
    else
        % Compute the number of nodal domains of each eigenfunction
        [posneg,VertConn]=nodal_sets_count(FV,V_nd,1,10,VertConn);
        nd_c=sum(posneg');
    end
end

%%test if the value of posneg == 2 and choice the good eigenvectors 
V_nd = V_nd(:,find(nd_c==2));


%% change the sign w.r.t x,y,z
for i=1:size(V_nd,2)
    rx(:,:,i)=corrcoef(V_nd(:,i),FV.vertices(:,1));
    ry(:,:,i)=corrcoef(V_nd(:,i),FV.vertices(:,2));
    rz(:,:,i)=corrcoef(V_nd(:,i),FV.vertices(:,3));
end

rxx = squeeze(abs(rx(1,2,:)))';
ryy = squeeze(abs(ry(1,2,:)))';
rzz = squeeze(abs(rz(1,2,:)))';

%index of the max corrcoef
[maxX, idx] = max(rxx);
[maxY, idy] = max(ryy);
[maxZ, idz] = max(rzz);

%% test if indice diff
if idx==idy
    ryy(idy) = NaN;
    [~,idy2] = max(ryy);
    idy = idy2;
end

if (idx==idz) || (idy==idz)
    rzz(idz) = NaN;
    [~,idz2] = max(rzz);
    idz = idz2;
end
V_final = V_nd(:,[idx,idy,idz]);

% Reorient if necessary, to raise the +1/-1 ambiguity on the sign of
% eigenfunctions
for ii=1:3
    [~,indm]=min(V_final(:,ii));
    [~,indM]=max(V_final(:,ii));
    
    if FV.vertices(indm,indices(ii))>FV.vertices(indM,indices(ii))
        tmp=indm;
        indm= indM;
        indM=tmp;
        V_final(:,ii)=-V_final(:,ii);
    end
end

%% Mapping on sphere
sphFV.faces=FV.faces;
sphFV.vertices=V_final;
% sphFV.vertices=V(:,1:3);
sphFV.vertices=sphFV.vertices./sqrt(repmat(sum(sphFV.vertices.^2,2),1,3));
