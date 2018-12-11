#python bb.py 1.jpg &&
#python bb.py 2.jpg &&
#python bb.py 3.jpg &&
#python bb.py 4.jpg &&
#python bb.py 5.jpg 

#!/bin/bash
for i in {1..720}
do
	if [ $?==1 ];then
		python bb.py $i.jpg 
	fi
done
