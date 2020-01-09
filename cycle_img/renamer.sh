i=1; temp=$(mktemp -p .); for file in image*
do
mv "$file" $temp;
mv $temp $(printf "image_%0.3d.jpg" $i)
i=$((i + 1))
done
