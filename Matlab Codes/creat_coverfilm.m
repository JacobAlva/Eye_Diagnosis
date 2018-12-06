function coverfilm=creat_coverfilm(img)
         [h,w]=size(img);
         coverfilm=ones(h,w);
         for j=1:h
            for i=1:w
               if (img(j,i)==0)||(isnan(img(j,i)==1))
                   coverfilm(j,i)=0;
               end
            end
         end
end