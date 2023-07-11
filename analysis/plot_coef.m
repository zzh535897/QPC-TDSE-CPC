zrange=[-12 0];

if(exist('coef1','var')&&length(size(coef1))==3)
    vm=double(m0)+(0:1:double(nm-1));
    il=0:1:double(nl-1);
    coef_draw=squeeze(max(abs(coef1).^2));

    figure;
    imagesc(vm,il,log10(coef_draw));caxis(zrange); hold on;
    ax=gca;
    set(ax.XAxis,'tickdirection','none');
    set(ax.YAxis,'tickdirection','none');
    for jl=1:nl
        plot(linspace(min(vm)-0.5,max(vm)+0.5,100),-0.5+double(jl)*ones(1,100),'k-','LineWidth',0.5);
    end
    for jm=min(vm):max(vm)
        plot(-0.5+double(jm)*ones(1,100),linspace(-0.5,double(nl)-0.5,100),'k-','LineWidth',0.5);
    end
    colorbar;
    xlabel('$m$','Interpreter','latex');
    ylabel('$l-|m|$','Interpreter','latex');
    set(gca,'YDir','normal');
else
    error('No coefficient object found.');
end