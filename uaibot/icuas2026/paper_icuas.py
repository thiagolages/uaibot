import numpy as np
import uaibot as ub
import uaibot_cpp_bind as ub_cpp
import random
import matplotlib.pyplot as plt
import pickle
import os
import argparse


# def param_curve(t, no_agent):
    
#     radius = 1
    
    
#     s = (2*np.pi)*t
     
#     axis_x = np.matrix([0,0,1]).T
#     axis_y = np.matrix([np.cos(s),np.sin(s),0]).T
#     axis_z = np.matrix([np.sin(s),-np.cos(s),0]).T
    
#     phi = [s+i*2*np.pi/no_agent for i in range(no_agent)]
#     pos_center = np.matrix([3*np.cos(s),3*np.sin(s),3.5]).T

#     q_tg = []
#     for i in range(no_agent):
#         q_tg.append(pos_center + radius*(np.cos(phi[i])*axis_x + np.sin(phi[i])*axis_y))

#     return q_tg

def param_curve(t, no_agent):
    
    radius = 1
    
    
    s = (2*np.pi)*t
     
    axis_x = np.matrix([np.cos(s), np.sin(s),0]).T
    axis_y = np.matrix([-np.sin(s), np.cos(s),0]).T
    axis_z = np.matrix([0,0,1]).T
    
    phi = [s+i*2*np.pi/no_agent for i in range(no_agent)]
    pos_center = np.matrix([3*np.cos(s),3*np.sin(s),3.5]).T

    q_tg = []
    for i in range(no_agent):
        q_tg.append(pos_center + radius*(np.cos(phi[i])*axis_x + np.sin(phi[i])*axis_z))

    return q_tg
        
def create_curve(no_agent, no_points):

    q_tg = []

    for k in range(no_points):
        s = k/(no_points)
        q_tg.append(param_curve(s,no_agent)) 
            

    return q_tg
            
       
def mov_obstacles_traj(t, no_moving_obstacles=False):
    
    if no_moving_obstacles:
        return [[np.matrix([0.0, 0.0, 0.0]).T, np.matrix([0.0, 0.0, 0.0]).T, np.matrix([0.0, 0.0, 0.0]).T]]
        
    r1 = 4
    w1 = 0.75*1/8
    p1 = np.matrix([r1*np.sin(w1*t), 0.0, 3.5]).T
    v1 = np.matrix([r1*w1*np.cos(w1*t), 0.0, 0.0]).T 
    a1 = np.matrix([-r1*w1*w1*np.sin(w1*t), 0.0, 0.0]).T 
 
    r2 = 4
    w2 = 0.75*1/4
        
    p2 = np.matrix([0, r2*np.cos(w2*t), 3.5]).T
    v2 = np.matrix([0, -r2*w2*np.sin(w2*t), 0.0]).T 
    a2 = np.matrix([0, -r2*w2*w2*np.cos(w2*t), 0.0]).T 
 
    r3 = 4
    w3 = 0.75*1/4
    c = np.sqrt(2)/2
        
    p3 = np.matrix([c*r3*np.cos(w3*t), c*r3*np.cos(w3*t), 4.5]).T
    v3 = np.matrix([-w3*c*r3*np.sin(w3*t), -w3*c*r3*np.sin(w3*t), 0.0]).T 
    a3 = np.matrix([-w3*w3*c*r3*np.cos(w3*t), -w3*w3*c*r3*np.cos(w3*t), 0.0]).T 
       
    return [[p1,v1,a1],[p2,v2,a2],[p3,v3,a3]]


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_name", type=str, default=None)
    parser.add_argument("-n", "--no_moving_obstacles", action="store_true")
    args = parser.parse_args()
    

    if args.save_name is None:
        raise ValueError("save_name is required")

    ##############################
    #Parameters
    ##############################

    param_no_agent = 6
    param_no_points = 3000
    param_dt = 0.05 #0.3 0.1 0.2
    param_tauv = 0.2
    param_Kc = 3
    param_Kt = 0.1*param_no_agent 
    param_h = 4.0

    param_radius_agent = 0.05
    param_height_agent = 0.4
    param_r = 0.2
    param_eps_d = 0.01
    param_safe_dist = 1.5
    param_eta=0.5
    param_dist_delta=0.003 #0.005
    param_moving_obstacle_radius = 0.3+0*0.45
    param_time_sim = 80 #20+0*120 

    colors=['red','green','blue','yellow','magenta','cyan','white','black']

    #Initial configuration
    q = []

    lst = list(range(param_no_agent))
    # random.shuffle(lst)

    for i in range(param_no_agent):
        theta = 2*np.pi*lst[i]/param_no_agent
        q.append(np.matrix([np.cos(theta), np.sin(theta),param_height_agent/2+0.05]).T)

    #Initial velocity
    dotq = [np.matrix(np.zeros((3,1))) for i in range(param_no_agent)]

    #Obstacles

    mesh_material_wood = ub.MeshMaterial.create_wood()
    mesh_material_metal = ub.MeshMaterial.create_rough_metal()

    obs0 = ub.Box(htm = ub.Utils.trn([0.,0.,0]),width=4.0,depth=4.0,height=0.02,mesh_material=mesh_material_wood)

    obs1 = ub.Box(htm = ub.Utils.trn([-2,0,0.5]),width=0.3,depth=4,height=1.0,mesh_material=mesh_material_wood)
    obs2 = ub.Box(htm = ub.Utils.trn([2,0,0.5]),width=0.3,depth=4,height=1.0,mesh_material=mesh_material_wood)
    obs3 = ub.Box(htm = ub.Utils.trn([0,-2,0.5]),width=4,depth=0.3,height=1.0,mesh_material=mesh_material_wood)
    obs4 = ub.Box(htm = ub.Utils.trn([0,2,0.5]),width=4,depth=0.3,height=1.0,mesh_material=mesh_material_wood)

    obs5 = ub.Box(htm = ub.Utils.trn([0,-4/3,1.0]),width=4.0,depth=0.3,height=0.15,mesh_material=mesh_material_metal)
    obs6 = ub.Box(htm = ub.Utils.trn([0,0,1.0]),width=4.0,depth=0.3,height=0.15,mesh_material=mesh_material_metal)
    obs7 = ub.Box(htm = ub.Utils.trn([0,4/3,1.0]),width=4.0,depth=0.3,height=0.15,mesh_material=mesh_material_metal)

    obs8 = ub.Box(htm = ub.Utils.trn([-4/3,0,1.0]),width=0.3,depth=4.0,height=0.15,mesh_material=mesh_material_metal)
    obs9 = ub.Box(htm = ub.Utils.trn([0,0,1.0]),width=0.3,depth=4.0,height=0.15,mesh_material=mesh_material_metal)
    obs10 = ub.Box(htm = ub.Utils.trn([4/3,0,1.0]),width=0.3,depth=4.0,height=0.15,mesh_material=mesh_material_metal)


    all_obstacles = [obs0,obs1,obs2,obs3,obs4,obs5,obs6,obs7,obs8,obs9,obs10]
    #all_obstacles = [obs0]

    #########################################################


    q_tg = create_curve(param_no_agent, param_no_points)

    all_obstacles_cpp = [obs.cpp_obj for obs in all_obstacles]

    a_fun = lambda _q, _dotq, _t: ub_cpp.icuas_gvf_acc(q = _q, 
                                                dotq = _dotq, 
                                                obstacles = all_obstacles_cpp,
                                                moving_obstacles = mov_obstacles_traj(_t, args.no_moving_obstacles),
                                                q_tg = q_tg, 
                                                Kc=param_Kc, 
                                                Kt=param_Kt, 
                                                h=param_h, 
                                                tau_v=param_tauv, 
                                                agent_radius = param_radius_agent,
                                                agent_height = param_height_agent,
                                                r = param_r,
                                                eps_d = param_eps_d,
                                                safe_dist = param_safe_dist,
                                                eta=param_eta,
                                                dist_delta = param_dist_delta,
                                                moving_obstacle_radius = param_moving_obstacle_radius,
                                                eps=1e-30)



    v_fun = lambda _q,: ub_cpp.icuas_gvf_vel(_q, q_tg, Kc=param_Kc, Kt=param_Kt, h=param_h, eps=1e-30)




    #sim = ub.Simulation.create_sim_mountain([])
    sim = ub.Simulation([])

    agents = []
    agents_model = []
    all_moving_obstacles = []
    no_moving_obs = len(mov_obstacles_traj(0, args.no_moving_obstacles))
    print("no_moving_obs = " + str(no_moving_obs))

    for i in range(param_no_agent):
        agents.append(ub.Cylinder(radius=param_radius_agent,height=param_height_agent,color=colors[i],opacity=0.3))
        agents_model.append(ub.RigidObject([ub.Model3D(url="https://raw.githubusercontent.com/viniciusmgn/uaibot_content/master/contents/CrazyFlie/crazyflie.obj",htm=ub.Utils.rotx(np.pi/2),scale=0.55)]))

    for i in range(no_moving_obs):
        all_moving_obstacles.append(ub.Ball(radius=param_moving_obstacle_radius, color='gray'))
        
    sim.add(agents)
    sim.add(agents_model)
    sim.add(all_obstacles)
    if not args.no_moving_obstacles:
        sim.add(all_moving_obstacles)
    
    hist_q=[]
    hist_dotq=[]
    hist_a = []
    hist_min_dist = []
    hist_t=[]
    hist_feasible=[]

    k_max = round(param_time_sim/(0.1*param_dt))
            

    for k in range(k_max):
        
        t = 0.1*k*param_dt
        out = a_fun(q,dotq,t)
        
        if out.min_dist_obs<0.05 and t>20:
            hh=0
        
        percentage = round(100*k/k_max)
        
        hist_t.append(t)
        hist_feasible.append(out.feasible)
        hist_q.append([np.matrix(_v) for _v in q])
        hist_dotq.append([np.matrix(_v) for _v in dotq])
        
        

        
        print(str(percentage) + "%, D = " + str(round(out.D,3)), end='\r')
        hist_min_dist.append(out.min_dist_obs)

        hist_a.append([np.matrix(_v) for _v in out.vec[i]])
    
        for i in range(param_no_agent):
            q[i] += param_dt*dotq[i]
            dotq[i] += param_dt*(np.matrix(out.vec[i]).T)
        
            agents[i].add_ani_frame(t,ub.Utils.trn(q[i]))
            agents_model[i].add_ani_frame(t,ub.Utils.trn(q[i]))
            
        all_mov_obstacles_traj = mov_obstacles_traj(t, args.no_moving_obstacles)
        
        for i in range(no_moving_obs):
            all_moving_obstacles[i].add_ani_frame(t,ub.Utils.trn(all_mov_obstacles_traj[i][0]))
        

    ##

    data=[hist_t, hist_q, hist_dotq, hist_a, hist_min_dist, hist_feasible]

    # save data
    cwd = os.getcwd()
    

    # save sim
    # Determine the target directory where the sim will be saved
    target_dir = cwd

    # Count the number of .html files in the target directory
    html_files = [f for f in os.listdir(target_dir) if f.endswith('.html')]
    html_count = len(html_files)

    # Pad with leading zeros (assume 2 digits, up to 99 files)
    prefix = f"{html_count:02d}_"

    # Prepend the number to the save_name
    numbered_save_name = prefix + args.save_name

    # Save the simulation with the new save name
    sim.save(target_dir, numbered_save_name)

    with open(os.path.join("data/", numbered_save_name + "_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    print("Simulation saved as " + numbered_save_name + ".html")
    print("Data saved as " + numbered_save_name + "_data.pkl")
    # sim.run()
        
        
if __name__ == "__main__":
    main()