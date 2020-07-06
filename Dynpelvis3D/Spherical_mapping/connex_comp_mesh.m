function [nbr]=connex_comp_mesh(FV,VertConn,seuil)


%/----Script Authors------------\
%|                              |
%|       J.Lefèvre, PhD         |
%|   julien.lefèvre@univ-amu.fr |
%|                              |
%\------------------------------/

%ii0=find(seuil==1);
visited=seuil;
nbr=0;

while sum(visited(:))>0
   ii=find(visited==1);
   visited=find_neighbours2(ii(1),VertConn,visited,FV);
   nbr=nbr+1;
    %sum(visited(:))
end


function [tovisit]=find_neighbours2(seed,VertConn,visited,FV)

tovisit=visited;
list=seed;

bool=0;
seed_n=VertConn{seed};
for ii=1:length(VertConn{seed})
    bool=tovisit(seed_n(ii))|bool;
end
if bool==0
   tovisit(seed)=0; 
end

while bool
    list_n=[];
    bool=0;
    for ii=1:length(list)
        tovisit(list(ii))=0;
        for kk=VertConn{list(ii)}
            if tovisit(kk)
                list_n=[list_n,kk];
                bool=bool|tovisit(kk);
                tovisit(kk)=0;
            end
        end
        
    end
    %view_surface('',FV.faces,FV.vertices,tovisit);
    list=list_n;
    %length(list)
end

return


