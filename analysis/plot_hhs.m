dipole_type=1;  %1=dipole moment, 2=dipole accelaration
xaxis_type=2; %1=Energy, 2=Harmonic Order

tmin=2*110;
tmax=8*110;
window=(t_list>tmin).*(t_list<tmax).*sin(pi*(t_list-tmin)/(tmax-tmin)).^2;

dip_cell={};
lab_cell={};
omg_cell={};
if(exist('z_list','var'))
    dip_cell=[dip_cell,-z_list];% charge=-1
    lab_cell=[lab_cell,'$S_z(\omega)$(a.u.)'];
    if(xaxis_type==2)
        omg_cell=[omg_cell,fieldz_0(2)];
    end
elseif(exist('x_list','var'))
    dip_cell=[dip_cell,-x_list];
    lab_cell=[lab_cell,'$S_x(\omega)$(a.u.)'];
    if(xaxis_type==2)
        omg_cell=[omg_cell,fieldx_0(2)];
    end
elseif(exist('y_list','var'))
    dip_cell=[dip_cell,-y_list];
    lab_cell=[lab_cell,'$S_y(\omega)$(a.u.)'];
    if(xaxis_type==2)
        omg_cell=[omg_cell,fieldy_0(2)];
    end
else
    error('No dipole objects found.');
end

for i=1:length(dip_cell)
    dip=dip_cell{i};
    lab=lab_cell{i};
    n_time=double(length(dip));
    n_half=ceil(n_time/2);

    dw=(2*pi/para_dt/n_time);
    w_freq=(1:n_half)*dw -dw;
    d_freq=fft(dip.*window)*para_dt/(2*pi); %our FFT factor is 1/(2pi)
    d_freq=d_freq(1:n_half);

    if(dipole_type==1)
        hhgs=4*pi^2*w_freq.^(+2)/137.036^2.*abs(d_freq.').^2; %if by dipole moment
    elseif(dipole_type==2)
        hhgs=4*pi^2*w_freq.^(-2)/137.036^2.*abs(d_freq.').^2; %if by dipole accelaration
    else
        error('invalid dipole type flag.');
    end
    figure;
    if(xaxis_type==1)
        plot(w_freq,hhgs,'LineWidth',1.2);
        xlabel('$\omega $(a.u.)','Interpreter','latex');
    else
        plot(w_freq/omg_cell{i},hhgs,'LineWidth',1.2);
        xlabel('Harmonic Order','Interpreter','latex');
    end
    ylabel(lab,'Interpreter','latex');
    set(gca,'fontsize',13,'YScale','log');
    set(gcf,'position',[100,50,800,600]);
end
