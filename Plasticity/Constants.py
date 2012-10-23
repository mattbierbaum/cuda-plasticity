# These variables are used to write x, y, and z with quotes in shorthand for convenience purpose.
x='x'
y='y'
z='z'
xyz = [x,y,z]

permutation = {}
permutation[x,y] = (1, z)
permutation[y,x] = (-1, z) 
permutation[y,z] = (1, x)
permutation[z,y] = (-1, x) 
permutation[x,z] = (-1, y)
permutation[z,x] = (1, y) 
permutation[x,x] = permutation[y,y] = permutation[z,z] = (0., None)

perm = {}
for i in [x,y,z]:
    for j in [x,y,z]:
        for k in [x,y,z]:
            perm[i,j,k] = 0.
perm[x,y,z] = perm[y,z,x] = perm[z,x,y] = 1.
perm[x,z,y] = perm[z,y,x] = perm[y,x,z] = -1.

