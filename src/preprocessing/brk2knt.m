function t = brk2knt(breaks,mults)

%Written by: Gordon Pipa.

s = sum(mults);
if s==0
   t = [];
else
   li = length(breaks);
      % make sure there is a multiplicity assigned to each break,
      % and drop any break whose assigned multiplicity is not positive.
   if length(mults)~=li, mults = repmat(mults(1),1,li); s = mults(1)*li;
   else
      fm = find(mults<=0);
      if ~isempty(fm), breaks(fm)=[]; mults(fm)=[]; li = length(breaks); end
   end
   mm = zeros(1,s);
   mm(cumsum([1 reshape(mults(1:li-1),1,li-1)])) = ones(1,li);
   t = breaks(cumsum(mm));
end
