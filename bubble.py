import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import PoissonDisk

def temperature(ii):
    return 0.1*np.exp(-ii/300)

def prob_accept(score_new,score,ii):
    return np.exp((score_new-score)/temperature(ii))

num_dims = 2

# bubble polytope is Px<=q where q is optimized by point allocation
P = np.array([[1,0],[0,1],[-1,0],[0,-1]])
P = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
num_rays = np.size(P,0)
#print(P)
#print(num_rays)

space_size = 1

# points to exclude
num_obst = 30
obstacle_points = 2*space_size*(np.random.rand(num_dims,num_obst)-0.5)

# points to include
include_points = np.transpose(np.array([[0,0],[0.1,0],[0,0.1]])) # just the three for now, near the centre
Pincl = P@include_points

# evaluation points
num_eval = 5000
#eval_points = 2*space_size*(np.random.rand(num_dims,num_eval)-0.5)
#print(eval_points)
sampler = PoissonDisk(d=2, radius=0.015)
eval_points = 2*space_size*(np.transpose(sampler.random(num_eval))-0.5)
num_eval = np.size(eval_points,axis=1)
#print(eval_points)

# initialize q to include all eval points
Peval = P@eval_points
qmax = np.max(Peval, axis=1)
q = np.array(qmax)
#print(q)

# now reduce q with a greedy algorithm to exclude all obstacles
Pobst = P@obstacle_points
obst_alloc = [[] for _ in range(num_rays)]
for ii in range(num_obst):
    #print(Px)
    diff = q - Pobst[:,ii]
    #print(diff)
    alloc = np.argmin(diff)
    obst_alloc[alloc] += [ii]
    q[alloc] = min((q[alloc],Pobst[alloc,ii]))
print(obst_alloc)

def score_func(q):
    # score the polygon by how many eval points are inside
    #print(Peval.shape)
    #print(q.shape)
    qtiled = np.transpose(np.tile(q,(num_eval,1)))
    #print(qtest.shape)
    points_in = np.all(Peval<=qtiled,axis=0)
    #print(points_in)
    score = sum(points_in)/num_eval
    # TO DO add the inclusion points as a reward
    #####
    # add a regularizing term to favour larger q
    score += 0.001*np.sum(q/qmax)
    return(score,points_in)

score,points_in = score_func(q)

#plt.plot(space_size*[-1,1,1,-1,-1],space_size*[-1,-1,1,1,-1],'b-')
plt.plot(eval_points[0,:],eval_points[1,:],'k.')
plt.plot(eval_points[0,points_in],eval_points[1,points_in],'m.')
# special case for particular choice of P to make rectangle
#plt.plot([q[0],q[0],-q[2],-q[2],q[0]],[q[1],-q[3],-q[3],q[1],q[1]],'g-')

score_history = [score]
for ii in range(2000):
    # now try a random change
    choose_ray = np.random.choice([rr for rr in range(num_rays) if obst_alloc[rr]])
    #print(choose_ray)
    # find the closest obstacle allocated to that ray
    closest_obst_ix = np.argmin(Pobst[choose_ray,obst_alloc[choose_ray]])
    closest_obst = obst_alloc[choose_ray][closest_obst_ix]
    #print(closest_obst)
    #print(q)
    #print(Pobst[choose_ray,closest_obst])
    # move that obstacle to a random different ray
    other_ray = np.random.choice([rr for rr in range(num_rays) if rr != choose_ray])
    #print(other_ray)
    obst_alloc[choose_ray].remove(closest_obst)
    obst_alloc[other_ray].append(closest_obst)
    #print(obst_alloc)
    qnew = np.array(q)
    qnew[choose_ray] = np.min(Pobst[choose_ray,obst_alloc[choose_ray]],initial=qmax[choose_ray])
    qnew[other_ray] = min((qnew[other_ray],Pobst[other_ray,closest_obst]))
    #print('qnew',qnew)
    # just test that with a fresh calculation
    #qtest = np.array([np.min(Pobst[rr,obst_alloc[rr]],initial=qmax[rr]) for rr in range(num_rays)])
    #print('qtest',qtest)
    score_new,_ = score_func(qnew)
    #print(score_new)
    if score_new>score:
        # always accept improvement
        print('improve',score,score_new)
        q = qnew
        score = score_new
    elif np.random.random()<prob_accept(score_new,score,ii):
        # sometimes accept degradation
        print('accept',score,score_new)
        q = qnew
        score = score_new
    else:
        # undo change
        obst_alloc[choose_ray].append(closest_obst)
        obst_alloc[other_ray].remove(closest_obst)
        print('reject',score,score_new)
    score_history.append(score)

_,points_in = score_func(q)
plt.plot(eval_points[0,points_in],eval_points[1,points_in],'g.')
#plt.plot(eval_points[0,~points_in],eval_points[1,~points_in],'m.')
#plt.plot([q[0],q[0],-q[2],-q[2],q[0]],[q[1],-q[3],-q[3],q[1],q[1]],'c-')
plt.plot(obstacle_points[0,:],obstacle_points[1,:],'rs')
plt.plot(include_points[0,:],include_points[1,:],'gs')
plt.show()

plt.plot(score_history)
plt.show()
