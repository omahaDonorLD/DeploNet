#! /usr/bin/env python2
import sys
for i in range(1,len(sys.argv)) :
	namefile=sys.argv[i]
	f = open(namefile, 'r')
	tmp=[line[:-1] for line in f ]
	f.close()
	print(namefile)
	namefile="tities/gnplot"+namefile
	f=open(namefile,"w")
	j=1
	for k in tmp :
		if len(k) > 0 :
			f.write(str(j)+","+k+"\n")
			j=j+1
	f.close()
