import sys
if len(sys.argv) > 1 :
	res=[[] for i in range(len(sys.argv)-1)]
	for i in range(1,len(sys.argv)) :
		print (i, " -- ", sys.argv[i])
		f = open(sys.argv[i], 'r')
		tmp=[line[:-1] for line in f ]
		print(len(tmp))
		#print ([ [float(j) for j in ele.split(",")] for ele in tmp ])
		if i == 5 :
			f2 = open("hallal"+str(i), 'w')
			for ele in tmp :
				for subele in ele :
					f2.write(subele)
				f2.write("\n")
		f.close()
else :
	print("hello world")
