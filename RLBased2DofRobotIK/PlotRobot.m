function [] = PlotRobot(link1,link2,theta1,theta2)
Tee = eye(4,4);
theta = [theta1, theta2];
a = [link1, link2];
for i=1:size(theta,2)
    temp = Tee;
    T(:,:,i) = DHMatrixModify(0,a(i),0,theta(i));
    Tee = Tee * T(:,:,i);
    plot2([temp(1,4) Tee(1,4)],[temp(2,4) Tee(2,4)],[temp(3,4) Tee(3,4)],'k','LineWidth',1),hold on
    trplot(Tee,'frame',num2str(i),'thick',1,'rgb','length',50)
    xlabel('X-axis'),ylabel('Y-axis'),zlabel('Z-axis')
end
end