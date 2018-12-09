foo = potential;
foo.mesh = 1;
foo.lattice = [1:6; 8:-1:3; 1 3 5 9 8 3];
foo.wing = zeros(7,6);
foo.wing(1:2,:) = [1 1 1 1 1 1; -2 -1 2 3 4 5];
foo.wing(4,:) = 1.5;
foo.wing(5,:) = 5;
foo.wing(5,5) = 45;
foo.vrt = zeros(size(foo.lattice));
foo.hrz = zeros(size(foo.lattice));
foo.update()
