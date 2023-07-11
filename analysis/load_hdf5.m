function load_hdf5(filename,dst)
    if(length(filename)<=3||~strcmp(filename(end-2:end),'.h5'))
        filename=[filename,'.h5'];
    end
    if(nargin==1)
        dst='base';%otherwise, use 'caller'
    end
    info=h5info(filename);
    nvar=length(info.Datasets);
    for i=1:nvar
        name=info.Datasets(i).Name;
        assignin(dst,name,h5read(filename,['/',name]));
    end
end