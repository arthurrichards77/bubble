import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import PoissonDisk
from math import atan2

def poly_2d_P(n):
    # bubble polytope is Px<=q where q is optimized by point allocation
    P = np.transpose(np.array([np.cos(np.linspace(2*np.pi/n,2*np.pi,n)),
                               np.sin(np.linspace(2*np.pi/n,2*np.pi,n))]))
    return(P)

def plot_poly_2d(P,q,style='g-',tol=1e-6):
    # plot the polygon x in R^2, Px<=q
    # check sizes are 2D and compatible
    assert np.size(P,1) == 2
    num_rays = np.size(P,0)
    assert q.shape == (num_rays,)
    # start by identifying vertices at ray intersections
    vertices = []
    for ii in range(num_rays):
        for jj in range(ii):
            vertex = np.linalg.solve(P[[ii,jj],:],
                                     q[[ii,jj]])
            if np.all(P@vertex <= q*(1.0+tol)):
                vertices.append(vertex)
            #print(vertices)
    centroid = sum(vertices)/len(vertices)
    def angle_sort(v):
        return atan2(v[1] - centroid[1], v[0] - centroid[0])
    vertices.sort(key=angle_sort)
    plt.plot([v[0] for v in vertices] + [vertices[0][0]],
             [v[1] for v in vertices] + [vertices[0][1]], style)

class Bubble:

    def __init__(self, P, space_size=1.0):
        # key parameters are the "rays" P and the size
        self.P = P
        self.space_size = space_size
        # dimensions inferred from provided P
        self.num_rays,self.num_dims = P.shape
        print(f'Problem has {self.num_dims} dimensions')
        print(f'Polygon has {self.num_rays} edges')
        # set of points for evaluating approximate volume
        self.eval_points = self._eval_points_poisson()
        self.num_eval = np.size(self.eval_points,axis=1)
        print(f'Generated {self.num_eval} evaluation points')
        self._Peval = self.P@self.eval_points
        self.qmax = np.max(self._Peval, axis=1)
        # set of points to be inside the bubble
        self.include_points = None
        self.num_incl = 0
        self.qmin = -np.inf*np.ones_like(self.qmax)
        # set of points to be outside the bubble
        self.obstacle_points = None
        self.num_obst = 0
        self._Pincl = None
        self._Pobst = None
        # placeholders for solution
        self.obst_alloc = [[] for _ in range(self.num_rays)]
        self.q = None
        self.score = None
        self.score_history = []

    def _eval_points_poisson(self, n=None, radius=None):
        if radius is None:
            radius = self.space_size/30.0
        if n is None:
            n = int(self.space_size/radius)**self.num_dims
        print(f'Generating up to {n} Poison disk samples')
        sampler = PoissonDisk(d=self.num_dims, radius=radius)
        return(2*self.space_size*(np.transpose(sampler.random(n))-0.5))

    def set_include_points(self, include_points):
        assert np.size(include_points,0)==self.num_dims
        self.include_points = include_points
        self._Pincl = self.P@include_points
        self.num_incl = np.size(include_points,1)
        print(f'Received {self.num_incl} inclusion points')
        self.qmin = np.max(self._Pincl, axis=1)
        print('qmin',self.qmin)
        assert np.all(self._points_in(self._Pincl,self.qmax))

    def set_obstacle_points(self, obstacle_points):
        assert np.size(obstacle_points,0)==self.num_dims
        self.obstacle_points = obstacle_points
        self._Pobst = self.P@obstacle_points
        self.num_obst = np.size(obstacle_points,1)
        print(f'Received {self.num_obst} obstacle points')

    def init(self):
        # set q to its biggest useful extent
        #self.q = np.array(self.qmin)
        # enlarge q with a greedy algorithm
        for ii in range(self.num_obst):
            diff = self._Pobst[:,ii] - self.qmin
            alloc = np.argmax(diff)
            self.obst_alloc[alloc] += [ii]
        self.q = np.array([np.min(self._Pobst[rr,self.obst_alloc[rr]],initial=self.qmax[rr]) for rr in range(self.num_rays)])
        assert np.all(self.q>self.qmin)

    def q_from_alloc(self):
        # re-calculate q for Px<=q using the obstacle-to-ray allocations in obst_alloc
        # useful for verification or for recreating bubble from AI allocation
        q_now = np.array([np.min(self._Pobst[rr,self.obst_alloc[rr]],initial=self.qmax[rr]) for rr in range(self.num_rays)])
        return q_now

    def _temperature(self, ii):
        return 0.05*np.exp(-ii/300)

    def _prob_accept(self, score_new, score, ii):
        return np.exp((score_new-score)/self._temperature(ii))

    def _points_in(self, Ppoints, q):
        qtiled = np.transpose(np.tile(q,(np.size(Ppoints,1),1)))
        return np.all(Ppoints<=qtiled,axis=0)

    def score_func(self, q):
        points_in = self._points_in(self._Peval,q)
        score = sum(points_in)/self.num_eval
        # TO DO add the inclusion points as a reward
        #####
        # add a regularizing term to favour larger q
        score += 0.001*np.sum(q/self.qmax)
        return score

    def solve(self, num_iters=2000):
        self.score = self.score_func(self.q)
        self.score_history = [self.score]
        print(f'Initial score is {self.score}')
        for ii in range(num_iters):
            # now try a random change - choose a ray with non-zero obstacle allocation
            choose_ray = np.random.choice([rr for rr in range(self.num_rays) if self.obst_alloc[rr]])
            # find the closest obstacle allocated to that ray
            closest_obst_ix = np.argmin(self._Pobst[choose_ray,self.obst_alloc[choose_ray]])
            closest_obst = self.obst_alloc[choose_ray][closest_obst_ix]
            #print(closest_obst)
            #print(q)
            #print(Pobst[choose_ray,closest_obst])
            # move that obstacle to a random different ray
            other_ray = np.random.choice([rr for rr in range(self.num_rays) if rr != choose_ray])
            #print(other_ray)
            self.obst_alloc[choose_ray].remove(closest_obst)
            self.obst_alloc[other_ray].append(closest_obst)
            #print(obst_alloc)
            qnew = np.array(self.q)
            qnew[choose_ray] = np.min(self._Pobst[choose_ray,self.obst_alloc[choose_ray]],
                                      initial=self.qmax[choose_ray])
            qnew[other_ray] = min((qnew[other_ray],self._Pobst[other_ray,closest_obst]))
            #print('qnew',qnew)
            # just test that with a fresh calculation
            #qtest = np.array([np.min(Pobst[rr,obst_alloc[rr]],initial=qmax[rr]) for rr in range(num_rays)])
            #print('qtest',qtest)
            score_new = self.score_func(qnew)
            #print(score_new)
            if np.any(qnew<self.qmin):
                # discard change that removes inclusion point
                self.obst_alloc[choose_ray].append(closest_obst)
                self.obst_alloc[other_ray].remove(closest_obst)
            elif score_new>self.score:
                # always accept improvement
                #print('improve',score,score_new)
                self.q = qnew
                self.score = score_new
            elif np.random.random()<self._prob_accept(score_new,self.score,ii):
                # sometimes accept degradation
                #print('accept',score,score_new)
                self.q = qnew
                self.score = score_new
            else:
                # undo change
                self.obst_alloc[choose_ray].append(closest_obst)
                self.obst_alloc[other_ray].remove(closest_obst)
                #print('reject',score,score_new)
            self.score_history.append(self.score)
        print(f'Finished with score {self.score}')
        print('P=',self.P)
        print('q=',self.q)

    def plot_2d_result(self, point_style='g.', poly_style='g-'):
        assert self.num_dims==2
        points_in = self._points_in(self._Peval,self.q)
        plt.plot(self.space_size*np.array([-1,1,1,-1,-1]),
                 self.space_size*np.array([-1,-1,1,1,-1]),'b-')
        #plt.plot(self.eval_points[0,:],self.eval_points[1,:],'k.')
        plt.plot(self.eval_points[0,points_in],self.eval_points[1,points_in],point_style)
        plt.plot(self.obstacle_points[0,:],self.obstacle_points[1,:],'rs')
        plt.plot(self.include_points[0,:],self.include_points[1,:],'bs')
        plot_poly_2d(self.P,self.q,poly_style)
        #plt.show()

    def plot_solve_history(self):
        plt.plot(self.score_history)
        #plt.show()

def run_example():
    P = poly_2d_P(6)
    bubble = Bubble(P,1.0)
    # random obstacles
    obstacle_points = 2.0*(np.random.rand(2,30)-0.5)
    bubble.set_obstacle_points(obstacle_points)
    # token points inside
    include_points = np.transpose(np.array([[0,0],[0.1,0],[0,0.1]])) # just the three for now, near the centre
    bubble.set_include_points(include_points)
    # initialize and plot first solution
    bubble.init()
    bubble.plot_2d_result(point_style='m+',poly_style='m-')
    # solve
    bubble.solve()
    # verify q is OK
    assert np.all(bubble.q == bubble.q_from_alloc())
    print('q test passed OK')
    # plot
    bubble.plot_2d_result()
    plt.figure()
    bubble.plot_solve_history()
    plt.show()

if __name__=='__main__':
    run_example()
