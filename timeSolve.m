function timeSolve(n)
   m = rand(n);
   b = rand(n,1);
   tic
   linsolve(m,b);
   toc
end
