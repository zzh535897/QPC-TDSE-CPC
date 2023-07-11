
if(exist('lmd1','var'))
    figure;hold on;
    for im=1:nm
        for il=1:nl
            plot(pr.^2/2,squeeze(abs(lmd(im,il,:))).^2,'LineWidth',1.1);
        end
    end
    xlabel('$E$(a.u.)'        ,'Interpreter','latex');
    ylabel('$P_{lm}(E)$(a.u.)','Interpreter','latex');
    title('Partial-Wave PMD');

    figure;hold on;
    pes=sum(sum(abs(lmd).^2,1),2);
    plot(pr.^2/2,pes,'k-','LineWidth',1.1);

    xlabel('$E$(a.u.)'   ,'Interpreter','latex');
    ylabel('$P(E)$(a.u.)','Interpreter','latex');
    title('PES');
end