def a(rank, s, n):
	sum = 0
	for i in range(1, n+1):
		sum = sum + i**(-s)
	q = 1 / (rank**s * sum)
	return q


print(a(1,2,3))
print(a(2,2,3))
print(a(3,2,3))