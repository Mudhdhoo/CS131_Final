import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from skimage import io
from chan_vese import chan_vese

def updatefig(frame):
    im.set_array(phi_list[frame] > 0)
    return im,

def calc_seg_average(seg, im):
     avr_inside = np.average(seg*im)
     avr_outside = np.average((seg == 0)*im)
     
     return avr_outside, avr_inside

def dice_score(segmentation, ground_truth):
    accuracy = np.sum(segmentation==ground_truth)/(segmentation.size)
    return accuracy

# Import image
img = io.imread('images/mole.png', as_gray = True)
truth = np.load('results_mole/mole_truth.npy')
img_orig = img.copy()
m, n = img.shape

# Generate missing indices
percent_missing = 0.98
known = np.ones((m,n))
for i in range(m):
    for j in range(n):
        if np.random.random() > percent_missing:
                known[i:i+6, j:j+6] = 0

# Corrupt image
img = known * (img + np.random.normal(0,0.2,(m,n)))

# Evaluate raw Chan-Vese
seg_raw, phi = chan_vese(img, mu=0.2, init_level_set='checkerboard', max_num_iter=200)
dice_raw = dice_score(seg_raw, truth)

start = time.time()

# Initial Reconstruction
U = cp.Variable(shape=(m, n))
expr = cp.tv(U)
constraints = [cp.multiply(known, U) == cp.multiply(known, img)]
prob = cp.Problem(cp.Minimize(expr), constraints)
prob.solve(verbose=False, solver = cp.SCS, max_iters = 1000)
print("optimal objective value: {}".format(expr.value))
img_recon_init = U.value

# Initial segmentation
seg_init, phi = chan_vese(img_recon_init, mu=0.2, init_level_set='checkerboard', max_num_iter=200)

# Evaluate initial segmentation
dice_init = dice_score(~seg_init, truth)

# Loop
iter = 5
lambda1 = 0.0005
lambda2 = 0.0005
lmbda = 0.01
phi_list = phi
seg = seg_init
img_recon = img_recon_init
for i in range(iter):
    U = cp.Variable(shape=(m, n))
    avr_outside, avr_inside = calc_seg_average(seg, img_recon)

    expr = cp.norm(U-img_recon, 'fro') + lmbda*cp.tv(U) + lambda1*cp.sum_squares((cp.multiply(U,seg) - avr_inside)) + lambda2*cp.sum_squares((cp.multiply(U,(seg == 0)) - avr_outside))
    constr = []
    prob = cp.Problem(cp.Minimize(expr), constr)
    prob.solve(verbose=False, solver = cp.SCS, max_iters = 350)
    print("optimal objective value: {}".format(expr.value))
    img_recon = U.value

    seg, phi = chan_vese(img_recon, mu=0.1, init_level_set=seg, max_num_iter=3, u_n= 1*(phi_list[-1]))
    phi_list += phi

end = time.time()
print('Execution time:', end - start)

# Evaluate final segmentation
dice_final = dice_score(~seg, truth)

print('DICE Chan-Vese:', dice_raw)
print('DICE initial segmentation:', dice_init)
print('DICE final:', dice_final)

# Plotting and animation
fig = plt.figure()
im = plt.imshow(phi_list[0] > 0, animated=True)

ani = FuncAnimation(fig, updatefig,  blit=True, frames = np.arange(0,200,1))
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(10, 5))
axes[0,0].imshow(img_orig) 
axes[0,1].imshow(img)
axes[0,2].imshow(img_recon)
axes[1,0].imshow(truth)
axes[1,1].imshow(~seg)
axes[1,2].imshow(seg_raw)

plt.tight_layout()
plt.show()

# Save plots
fig = plt.figure()
plt.imshow(img_orig)
plt.axis('off')
fig.savefig('results_mole/original_mole.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(img)
plt.axis('off')
fig.savefig('results_mole/damaged_mole.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(img_recon)
plt.axis('off')
fig.savefig('results_mole/reconstructed_mole.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(truth)
plt.axis('off')
fig.savefig('results_mole/true_seg_mole.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(~seg)
plt.axis('off')
fig.savefig('results_mole/mole_seg.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(seg_raw)
plt.axis('off')
fig.savefig('results_mole/mole_chan_vese.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(~seg_init)
plt.axis('off')
fig.savefig('results_mole/mole_seg_init.png', bbox_inches='tight', pad_inches=0) 

fig = plt.figure()
plt.imshow(img_recon_init)
plt.axis('off')
fig.savefig('results_mole/mole_recon_init.png', bbox_inches='tight', pad_inches=0) 



