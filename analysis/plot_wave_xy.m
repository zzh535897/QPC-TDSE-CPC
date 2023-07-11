ith=ceil(disp_nt/2);

[p_ph,p_rd]=meshgrid(ph(1:end),rd);
wave_draw=abs(squeeze(wave1(:,ith,1:end))).^2;
figure;
pcolor(p_rd.*cos(p_ph),p_rd.*sin(p_ph),log10(wave_draw));
shading interp;
colorbar;colormap(jet);
caxis([-10 -0]);
xlabel('$x$','interpreter','latex');
ylabel('$y$','interpreter','latex');
set(gca,'FontSize',20);
set(gcf,'position',[100,50,940,750]);%(centerX,centerY,Width,Height)
clear wave_draw;