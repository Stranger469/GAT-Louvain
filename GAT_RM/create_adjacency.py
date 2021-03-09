import time
import numpy as np
from multiprocessing import Manager, Pool


def construct_adj(max_degree, num_nodes, data, adj_info, adj_dict, pid, start, end):
    '''
    Construct adj table
    '''
    missed = 0
    adj = num_nodes*np.ones((num_nodes+1, max_degree), dtype=np.int32)
    deg = np.zeros((num_nodes,))
    print('adj process {} start, from {} to {}'.format(pid, int(start), int(end)), flush=True)
    # print(adj, deg)
    # proof = True
    # proof_n = 0
    for nodeid in data.UserId.unique()[int(start):int(end)]:
        # print(nodeid, flush=True)
        # print(adj_dict)
        neighbors = np.array([neighbor for neighbor in 
                            adj_info.loc[adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
        # print(neighbors, flush=True)
        deg[nodeid] = len(neighbors)
        if len(neighbors) == 0:
            missed += 1
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        # if proof:
            # proof_n = (nodeid, neighbors, deg[nodeid])
            # proof = False
        adj[nodeid] = neighbors
    
    adj_dict[pid] = (adj, deg)
    # adj_dict['proof_{}'.format(pid.split('_')[1])] = proof_n
    # print('Unexpected missing during constructing adj list: {}'.format(missed))
    print('{} complete'.format(pid), flush=True)

def construct_adj_multiProcess(data, num_nodes, max_degree, adj_info, num_process):
    '''
    Construct adj with multi procress optimized for 'num_process' cores CPU
    '''
    start_time = time.time()
    print('start multi procress processing')
    max_len = len(data.UserId.unique())

    # Initialized process
    start = []
    end = []
    processList = []
    min_max = []
    for i in range(num_process):
        start.append(max_len*i/num_process)
        end.append(max_len*(i + 1)/num_process)
        processList.append('adj_{}'.format(i + 1))
        min_max.append((
            data.UserId.unique()[int(max_len*i/num_process):int(max_len*(i + 1)/num_process)].min(), 
            data.UserId.unique()[int(max_len*i/num_process):int(max_len*(i + 1)/num_process)].max() + 1
        ))
    print(min_max)
    index_num = 0
    adj_dict = Manager().dict()
    # Start process pool
    pool = Pool(num_process)
    for pid in processList:
        # Start process for adjacency matrix construction
        pool.apply_async(func=construct_adj, 
                         args=(max_degree, num_nodes, data, adj_info, 
                            adj_dict, pid, start[index_num], end[index_num]))
        index_num += 1
    # close process and wait for all process end
    pool.close()
    pool.join()
    # print(adj_dict)

    # marge results
    adj_merge = num_nodes*np.ones((num_nodes+1, max_degree), dtype=np.int32)
    deg_merge = np.zeros((num_nodes,))
    for i in range(num_process):
        min = min_max[i][0]
        max = min_max[i][1]
        adj_merge[min:max] = adj_dict['adj_{}'.format(i+1)][0][min:max]
        deg_merge[min:max] = adj_dict['adj_{}'.format(i+1)][1][min:max]
    
    end_time = time.time()
    print('All adj done. time={}, process={}'.format(end_time - start_time, num_process), flush=True)
    # TEST
    # for i in range(num_process):
    #     min = min_max[i][0]
    #     print(adj_dict['proof_{}'.format(i+1)][1] == adj_merge[adj_dict['proof_{}'.format(i+1)][0]])
    #     print(adj_dict['proof_{}'.format(i+1)][2] == deg_merge[adj_dict['proof_{}'.format(i+1)][0]])
    return adj_merge, deg_merge
