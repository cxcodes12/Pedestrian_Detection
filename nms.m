function top = nms(boxes, overlap,s)

if isempty(boxes)
  top = [];
  return;
end

x1 = boxes(:,1);
y1 = boxes(:,2);
x2 = x1+boxes(:,3);
y2 = y1+boxes(:,4);


area = (x2-x1+1) .* (y2-y1+1);
[vals, I] = sort(s,'ascend');

pick = s*0;
counter = 1;
while ~isempty(I)
  
  last = length(I);
  i = I(last);  
  pick(counter) = i;
  counter = counter + 1;
  
  xx1 = max(x1(i), x1(I(1:last-1)));
  yy1 = max(y1(i), y1(I(1:last-1)));
  xx2 = min(x2(i), x2(I(1:last-1)));
  yy2 = min(y2(i), y2(I(1:last-1)));
  
  w = max(0.0, xx2-xx1+1);
  h = max(0.0, yy2-yy1+1);
  
  o = w.*h ./ area(I(1:last-1));
  
  I([last; find(o>overlap)]) = [];
end

pick = pick(1:(counter-1));
top = boxes(pick,:);

end