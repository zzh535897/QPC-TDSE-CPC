if(exist('fx_list','var') && exist('fy_list','var'))
    figure;hold on;
    
    xmax=max(fx_list);ymax=max(fy_list);
    xmin=min(fx_list);ymin=min(fy_list);
    xabsmax=max(abs(xmax),abs(xmin));
    yabsmax=max(abs(ymax),abs(ymin));
    absmax=max(xabsmax,yabsmax);

    plot(-fx_list,-fy_list,'k','LineWidth',1.1); %plot -A
    xlabel('$-A_x$(a.u.)','Interpreter','latex');
    ylabel('$-A_y$(a.u.)','Interpreter','latex');
    xlim([-absmax,+absmax]);
    ylim([-absmax,+absmax]);
    box on;
    
end