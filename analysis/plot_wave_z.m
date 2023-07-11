logscale = 1; %1= enable, 0=disable
fill_lower =1; %1 = enable, 0=disable 
iph=0;

if(exist('wave1','var'))
    wave_draw=abs(wave1).^2;
else
    error('No wave object found');
end
if(iph>=disp_np)
    error('too large iph.');
end
if(logscale==1)
    wave_draw=log10(wave_draw);
    crange=[-10 0];
else
    crange=[0 1];
end
if(fill_lower==1)
    [p_th,p_rd]=meshgrid([-th(end:-1:2),th],rd);
    wave_draw=[squeeze(wave_draw(:,end:-1:2,iph+1)),squeeze(wave_draw(:,:,iph+1))];
else
    [p_th,p_rd]=meshgrid(th,rd);
    wave_draw=squeeze(wave_draw(:,:,iph+1));
end
    
    figure;
    pcolor(p_rd.*cos(p_th), p_rd.*sin(p_th),wave_draw);shading interp;
    caxis(crange);colorbar;colormap(jet);
    xlabel('$r$cos$(\theta)$','interpreter','latex');
    ylabel('$r$sin$(\theta)$','interpreter','latex');
    set(gca,'FontSize',22);
    set(gcf,'position',[100,50,940,750]);%(centerX,centerY,Width,Height)
    clear wave_draw p_rd p_th;
