normalization = 0; %1= E, 0=P
logscale = 1; %1= enable, 0=disable
fill_lower =1; %1 = enable, 0=disable 

ipp=ceil(npp/2); 

if(exist('pmd1','var'))
    pmd_draw=abs(pmd1).^2;
    tit_name='PMD-PCS';
elseif(exist('pmd2','var'))
    pmd_draw=abs(pmd2).^2;
    tit_name='PMD-tSURFF';
else
    error('No PMD object fould');
end

if(logscale==1)
    pmd_draw=log10(pmd_draw);
    crange=[-8 0];
else
    crange=[0 1];
end

if(fill_lower==1)
    [g_pr,g_pt]=meshgrid(pr,[-pt(end:-1:2);pt]);
    pmd_draw= [squeeze(pmd_draw(ipp,end:-1:2,:));squeeze(pmd_draw(ipp,:,:))];
else
    [g_pr,g_pt]=meshgrid(pr,pt);
    pmd_draw= squeeze(pmd_draw(ipp,:,:));
end

if(normalization==1)
    pmd_draw=pmd_draw.*g_pr;
end

    figure;
    pcolor(g_pr.*cos(g_pt),g_pr.*sin(g_pt),pmd_draw);
    caxis(crange);shading interp;colormap(jet);colorbar;

    xlabel('$p$cos($\theta$)','interpreter','latex');
    ylabel('$p$sin($\theta$)','interpreter','latex');
    title(tit_name);
    set(gca,'FontSize',22);
    set(gcf,'position',[100,50,940,750]);%(centerX,centerY,Width,Height)
    clear pmd_draw g_pr g_pt;