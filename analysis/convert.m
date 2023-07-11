%--------------------------------------------------------------------------
%----------------------------RAW DATA PRE-PROCESS--------------------------
%--------------------------------------------------------------------------
%reshape wave functions
wave_list={'wave0','wave1'};
for id_name=wave_list
    name=id_name{1};
    if(exist(name,'var'))
        eval([name,'=complex(',name,'(1:2:end),',name,'(2:2:end));']);
        eval([name,'=reshape(',name,',[disp_nr,disp_nt,disp_np]);']);
    end
end
clear wave_list id_name name;
%reshape coefficients
coef_list={'coef0','coef1'};
for id_name=coef_list
    name=id_name{1};
    if(exist(name,'var'))
        eval([name,'=complex(',name,'(1:2:end),',name,'(2:2:end));']);
        eval([name,'=reshape(',name,',[nr,nl,nm]);']);
    end
end
clear coef_list id_name name;
%prepare r-space data
if(exist('disp_nr','var'))
    if(exist('disp_ri','var'))
        disp_rmin=disp_ri;
        disp_rmax=disp_rf;
    end
    rd=linspace(disp_rmin,disp_rmax,disp_nr);
    th=linspace(disp_thmin,disp_thmax,disp_nt);
    ph=linspace(disp_phmin,disp_phmax,disp_np);
end
%prepare p-space data
if(exist('pmd1','var'))
    npp=length(pp);
    npt=length(pt);
    npr=length(pr);
    pmd1=reshape(complex(pmd1(1:2:end),pmd1(2:2:end)),[npp,npt,npr]);%spectrum=reshape(complex(spectrum(1:2:end),spectrum(2:2:end)),[npt,npr]);
end
if(exist('pmd2','var'))
    npp=length(pp);
    npt=length(pt);
    npr=length(pr);
    pmd2=reshape(complex(pmd2(1:2:end),pmd2(2:2:end)),[npp,npt,npr]);%spectrum=reshape(complex(spectrum(1:2:end),spectrum(2:2:end)),[npt,npr]);
end
if(exist('pad1','var'))
    pad_npr=length(pad_pr);
    pad_npt=length(pad_pt);
    pad1=reshape(complex(pad1(1:2:end),pad1(2:2:end)),[pad_npt,pad_npr]);
end
if(exist('lmd1','var'))
    npr=length(pr);
    lmd1=complex(lmd1(1:2:end),lmd1(2:2:end));
    lmd1=reshape(lmd1,[nm,nl,npr]);
end
%prepare projection
if(exist('proj_nnz','var'))
    n_proj=sum(proj_nnz(:));
    proj_val=complex(proj_val(1:2:end),proj_val(2:2:end));

    if(length(proj_val)==nr*nl*nm) %time-indep
        proj_val=reshape(proj_val,[nr,nl,nm]);
        proj_nnz=reshape(proj_nnz,   [nl,nm]);

        nrmp=sqrt(sum(sum(sum(abs(proj_val).^2))));
        ioyd=1-sum(sum(sum(abs(proj_val).^2))); % only for the correct filter, this is yield
    else %time-dep
        proj_val=reshape(proj_val,[n_proj,length(proj_val)/n_proj]);%2D array for c_{nlm}(t)
    end
end