normalization = 1; %1= E, 0=P
ipt=ceil(npt/2);

logscale = 0; %1= enable,0=disable

    if(exist('pmd1','var'))
        pmd_draw = abs(squeeze(pmd1(:,ipt,:))).^2;
        tit_name='PMD-PCS';
    elseif(exist('pmd2','var'))
        pmd_draw = abs(squeeze(pmd2(:,ipt,:))).^2;
        tit_name='PMD-tSURFF';
    else
        error("no PMD object found.");
    end
    [g_pr,g_pp]=meshgrid(pr,pp);
    figure;
    
    if(normalization==1)%if you like E-normalization
        pmd_draw=pmd_draw.*g_pr;
    else%or you prefer P-normalization
        pmd_draw=pmd_draw*1.0;
    end

    if(logscale==1)%fine structures is only visible in logscale 
        pmd_draw=log10(pmd_draw);
    end
    pcolor(g_pr.*cos(g_pp),g_pr.*sin(g_pp),pmd_draw);
    if(logscale==0)
        caxis([ 0 1]);shading interp;colormap(jet);colorbar;
    else
        caxis([-8 0]);shading interp;colormap(jet);colorbar;
    end
    xlabel('$p\cos(\varphi)$','interpreter','latex');
    ylabel('$p$sin($\varphi$)','interpreter','latex');
    title(tit_name);
    set(gca,'FontSize',20);
    set(gcf,'position',[100,50,860,680]);
