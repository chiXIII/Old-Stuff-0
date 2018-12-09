res = 50; x1 = linspace(0,1,res);
x1 = x1(1:end-1);
x2 = linspace(1,1,res);
x2 = x2(1:end-1);
x3 = linspace(1,0,res);
x3 = x3(1:end-1);
x4 = linspace(0,0,res);
x4 = x4(1:end-1);
x = [x1 x2 x3 x4];
y = [x4 x1 x2 x3];
axis('square')
%{
for xi = [0.25 0.75]
	for yi = [0.25 0.75]
		plotSimple(xi,yi)
	end
end
%}
hold on
plot(x,y)

for xi = linspace(0,1,10)
	for yi = linspace(0,1,10)
		plotSimple(xi,yi)
	end
end

function s = simple(x,y)
	s = zeros(2,4);
	for i = 1:4
		x0 = mod(i+1,2);
		y0 = ceil(i/2)-1;
		r = [x ; y] - [x0 ; y0];
		r = r * (-1)^(ceil(i/2));
		s(:,i) = r/sum(r.^2);
	end
	s = sum(s,2)*0.05;
end

function plotSimple(x,y)
	hold on
	scatter(x,y);
	s = simple(x,y);
	s = s + [x ; y];
	plot([x s(1)], [y s(2)]);
	hold off
end
