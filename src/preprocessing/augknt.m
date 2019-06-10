function [augknot,addl] = augknt(knots,k,mults)

%Written by: Gordon Pipa.

if nargin<3
   if (length(k)>1|k<1)
      error('SPLINES:AUGKNT: wrong', ...
            'The second argument should be a single natural number.'), end
   mults = 1;
end

dk = diff(knots);
if ~isempty(find(dk<0)), knots = sort(knots); dk = diff(knots); end

augknot=[];
j=find(dk>0); if isempty(j)
   error('SPLINES:AUGKNT: too few knots', ...
         'The knot sequence should contain more than one point.'), end
addl = k-j(1);

interior = (j(1)+1):j(end);
%   % make sure there is a multiplicity assigned to each interior knot:
if length(mults)~=length(interior), mults = repmat(mults(1),size(interior)); end

augknot = brk2knt(knots([1 interior end]), [k mults k]);
