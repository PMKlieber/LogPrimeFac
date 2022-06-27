

# Pre-compute list for first n=1000 prime numbers
p=[]
for j in range(2,10000):
    i=j
    for k in p:
        if i%k==0:
            i=i/k
    if i==j:
        p.append(i)
        print(i)
	
import numpy as np

# Factor an integer into its prime factors
def primefac(i):
    # Iterate through known prime numbers (up to sqrt(i)), if its possible to divide by that factor, do so 
    # (as many times as possible), and move on until its fully factord
    r=[]
    j=i
    for k in p:
        while j%k==0:
            j=j/k
            r.append(k)
        if j==1 or k>i: return r
    return r



def rect(x,y,w,h,cl='cslr'):
   if cl=='cslr':
     col=clsr(w)
   elif cl=="calcol":
     col=calcol(list(p).index(int(np.round(sp.exp(w))))%len(cols))
   else:
     col='r'
   plt.fill((x,x+w,x+w,x),(y,y,y+h,y+h),c=col,edgecolor='black',linewidth=1)


cols=['r','y','g','c','b','m','silver','gray','brown']

cslr=lambda j:cols[list(p).index(int(np.round(sp.exp(j))))%len(cols)]
clsr=cslr 


def precs(n,y,txt=True):
  i=0
  for j in n:
	rect(i,y,j,1,cl='calcol')
	if txt: plt.text(i+j/2,y-.05,"ln {}".format(int(np.round(np.exp(j)))),va='bottom',size='small')
	i=i+j
		
cols=sp.vstack([colors.hsv_to_rgb((j,1,ffrac(k-1))) for j in sp.arange(0,6,1/(k))/6] for k in [1,2,3])            
cols= [colors.rgb2hex(i) for i in cols]  
calcol=lambda j: colors.hsv_to_rgb(((j%6)/6,1,ffrac(floor(j/6))))
calcol=lambda j:colors.hsv_to_rgb(((j%6)/6,0.34+0.66*(ceil((1+j)/6)%2),ffrac(floor(j/12))))

ffrac=lambda j: 1 if j ==0 else (1+2*(j-2**(floor(sp.log2(j)))))*(2**-(1+floor(sp.log2(j))))

for i in range(1,50):
	precs(np.log(pf(i)),i)


plt.show()





	import matplotlib.pyplot as plt
	import numpy as np
	import time


	def generate(X, Y, phi):
		"""
		Generates Z data for the points in the X, Y meshgrid and parameter phi.
		"""
		R = 1 - np.sqrt(X**2 + Y**2)
		return np.cos(2 * np.pi * X + phi) * R


	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	# Make the X, Y meshgrid.
	xs = np.linspace(-1, 1, 50)
	ys = np.linspace(-1, 1, 50)
	X, Y = np.meshgrid(xs, ys)

	# Set the z axis limits so they aren't recalculated each frame.
	ax.set_zlim(-1, 1)

	# Begin plotting.
	wframe = None
	tstart = time.time()
	for phi in np.linspace(0, 180. / np.pi, 100):
		# If a line collection is already remove it before drawing.
		if wframe:
			wframe.remove()

		# Plot the new wireframe and pause briefly before continuing.
		Z = generate(X, Y, phi)
		wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
		plt.pause(.001)

	print('Average FPS: %f' % (100 / (time.time() - tstart))) 