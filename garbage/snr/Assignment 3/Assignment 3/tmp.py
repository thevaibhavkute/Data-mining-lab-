
def freq(str):
	str = str.split()		
	str2 = []
	for i in str:			
		if i not in str2:
			str2.append(i)
	val=[]		
	for i in range(0, len(str2)):
		val.append(str.count(str2[i]))

    ans=""
    val.sort()
    for s in val:
        ans+=s+" "
    print(ans)

def main():
	str ='apple mango apple orange orange apple guava mango mango'
    k=3
	freq(str,k)					

if __name__=="__main__":
	main()			 # call main function
