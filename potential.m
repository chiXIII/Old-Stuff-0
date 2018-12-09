classdef potential < handle

   properties
      wing % x, y, z, chord, AOA, blockMask, reference
      mesh
      lattice % x y z magnitude
      freestream
      hrz
      vrt
   end

   methods

      function updateRef(self)
         s = size(self.lattice);
         diff1 = self.wing(4,:)*-0.5 .* cosd(self.wing(5,:));
         diff2 = self.wing(4,:)*-0.5 .* sind(self.wing(5,:));
         self.wing(:,:,7) = self.wing(1:3,:) + [diff1; zeros(1,s(2)); diff2];
         %{
         scatter(self.wing(1,:), self.wing(2,:));
         scatter(reference(1,:), reference(2,:));
         plot([1 1], [2 3]);
         %}
      end

      function setBlockMask(self)
         wingDiffs = self.wing(1:3,2:end) - self.wing(1:3,1:end-1);
         wing(:,:,6) = [modulus(wingDiffs) <= self.mesh false];
      end

      function updateHrzDiff(self)
         left = self.lattice(:, 1:end-1,1:3);
         right = self.lattice(:,2:end,1:3);
         diff = right - left;
         self.hrz(:,:,4:6) = diff(:,self.wing(:,1:end-1,6) );
      end

      function updateVrtDiff(self)
         upper = self.lattice(1:end-1,1:3);
         lower = self.lattice(2:end, 1:3);
         self.vrt(:,:,4:6) = lower - upper;
      end

      function updateHrzCenter(self)
         base = self.lattice(:,self.wing(:,:,6),1:3);
         self.hrz(:,:,1:3) = base + self.hrz(:,:,4:6)/2;
      end

      function updateVrtCenter(self)
         self.vrt(:,:,1:3) = self.lattice(:,1:end-1,1:3) + self.vrt(:,:,4:6)/2;
      end

      function update(self)
         self.updateRef();
         self.updateHrzDiff();
         self.updateVrtDiff();
         self.updateHrzCenter();
         self.updateVrtCenter();
      end

   end

end
