tmp=abs(pmd1).^2;

draw_th=squeeze(sum(sum(tmp,1),3)); draw_th=draw_th/max(draw_th); % renorm to 1.0
draw_ph=squeeze(sum(sum(tmp,2),3)); draw_ph=draw_ph/max(draw_ph); % renorm to 1.0

figure;
polarplot(pp,draw_ph','LineWidth',1.0);
thetalim([0,360]);
h=get(gca);

h.ThetaAxis.Label.String='$\varphi$ (deg)';
h.ThetaAxis.Label.Interpreter='latex';

figure;
polarplot(pt,draw_th','LineWidth',1.0);
thetalim([0,180]);

h=get(gca);

h.ThetaAxis.Label.String='$\theta$ (deg)';
h.ThetaAxis.Label.Interpreter='latex';
h.ThetaAxis.Label.Position=[-8 -1.15 0];

clear tmp;