from scipy.optimize import linprog
from armman.utils import prob_string_to_prob

def LP(lam_k, m_G, S_G, T, cluster_id, transition_probs):
    
    
    # S_G denotes the size of the group
    # S_G = 3 #For now harcoding it
    # m_G = 10 # the number of pulls we wish to satisfy. Hardcoding it
    # T = 20 # the number of time steps we have to pull
    # lam_k = 2 # this is the k-th Whittle Index per iteration, hard coding to 2

    # total number of variables is S_G + S_G*2 + 1 = 10
    #we have the following ordering of variables: v1>v2>v3>lambda_G>f10>f11>f20>f21>f30>f31 
    #c is for the coefficient of the (S_G + S_G*2 + 1 = 10) variables in the linear program
    c = [1]*S_G
    c.append(-m_G/T)
    for i in range(S_G*2):
        c.append(0)
    #c.reshape((4,1))

    #cluster_no for which we will define the group. We will use the transition probabilities of this group. For now 
#     cluster_no = cluster_transition_probabilities[cluster_transition_probabilities['cluster']== cluster_id] 

    #define the A matrix, which is of size (no. of variables x no. of equations). The no. of eqns is the size of b.
    A = [[0 for i in range(S_G + S_G*2 + 1)] for j in range(S_G*4)]
    for i in range(S_G*4):
        for j in range(S_G + S_G*2 + 1):
            if j==(i//4): #the coefficients of v's in the LP
                    A[i][j] = -1 
            if j==S_G and i%2==1:
                A[i][j] = 1 #the coefficient of lambda_G

            if (i%4) == 0:
                A[i][S_G+1+2*(i//4)] = -1 + prob_string_to_prob('P(NE, A, NE)', transition_probs[i%4])
                A[i][S_G+2+2*(i//4)] = prob_string_to_prob('P(NE, A, E)', transition_probs[i%4])

            elif (i%4) == 1:
                A[i][S_G+1+2*(i//4)] = -1 + prob_string_to_prob('P(NE, I, NE)', transition_probs[i%4])
                A[i][S_G+2+2*(i//4)] = prob_string_to_prob('P(NE, I, E)' , transition_probs[i%4])

            elif (i%4) == 2:
                A[i][S_G+1+2*(i//4)] = prob_string_to_prob('P(E, A, NE)', transition_probs[i%4])
                A[i][S_G+2+2*(i//4)] = -1 + prob_string_to_prob('P(E, A, E)', transition_probs[i%4])

            elif (i%4) == 3:
                A[i][S_G+1+2*(i//4)] = prob_string_to_prob('P(E, I, NE)', transition_probs[i%4])
                A[i][S_G+2+2*(i//4)] = -1 + prob_string_to_prob('P(E, I, E)', transition_probs[i%4])



    #define b vector is a vector of size S_G*4 = 12, that is there are 4 equations for each mother in group
    b = [0]*(S_G*4)
    for i in range(S_G*4):
        if i%4 == 0:
            b[i] = -1*lam_k
        elif i%4 == 1:
            b[i] = 0
        elif i%4 == 2:
            b[i] = -1 - lam_k
        elif i%4 == 3:
            b[i] = -1

    #define the bounds for the variables
    bound_for_variables = []
    for i in range(S_G + S_G*2 + 1):
        if i==S_G:
            bound_for_variables.append([0,None])
        else:
            bound_for_variables.append([None,None])

    #finally call linprog
    res = linprog(c, A_ub=A, b_ub=b, bounds=bound_for_variables, method='revised simplex')

    #retrieve the optimal value of lambda_G
    lambda_G_opt = res.x[S_G]
    return lambda_G_opt


def big_LP(m_C, ppl_per_cluster, T, k, transition_prob):
    
    # S_G denotes the size of the group
    # S_G = 3 #For now harcoding it
    # m_G = 10 # the number of pulls we wish to satisfy. Hardcoding it
    # T = 20 # the number of time steps we have to pull
    # lam_k = 2 # this is the k-th Whittle Index per iteration, hard coding to 2
     #number of beneficiaries

    N = ppl_per_cluster*len(m_C)

    cluster_ids = list(m_C.keys())


    # total number of variables is N + 1 + no_of_clusters + 2N = 123
    #we have the following ordering of variables: v1>. . .> v_N> lambda> lambda_C1 > lambda_C2 > f10>f11>f20> . . . > fN1 
     #c is for the coefficient of the (S_G + S_G*2 + 1 = 10) variables in the linear program
    c = [1]*N
    c.append(-(N-k))

    for i in cluster_ids:
        c.append(-m_C[i]/T)
    for i in range(N*2):
        c.append(0)

#     cluster_no = {}
#     for i in range(len(m_C)):
#         cluster_no[i] = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == cluster_ids[i]] 

    A = [[0 for i in range(3*N + 1 + len(m_C))] for j in range(N*4)]

    for i in range(N*4):
        for j in range(3*N+1+len(m_C)):
            if j==(i//4): #the coefficients of v's in the LP
                A[i][j] = -1 
            if j==N and i%2==0:
                A[i][j] = 1 #the coefficient of lambda
            if j == (N+1+(i//(ppl_per_cluster*4))) and i%2==1: 
                A[i][j] = 1

            #here transition_prob is a dictionary with N key-value pairs, each key containing the corresponding beneficiaries\
            #transition prob. The beneficiaries in transition_prob are ordered cluster-wise, i.e. beneficiaries from\
            #cluster_id[0] correspond keys 0 to ppl_per_cluster, then beneficiaries from cluster_id[1] correspond\
            #to keys ppl_per_cluster+1 + 2*ppl_per_cluster, and so on.
            #the current_trans_prob contains the present transition prob for the instant of the loop given by i//4.
            current_trans_prob = transition_prob[i//4]

            if (i%4) == 0:
                A[i][N+1+len(m_C)+2*(i//4)] = -1 + prob_string_to_prob('P(NE, A, NE)', current_trans_prob)
                A[i][N+2+len(m_C)+2*(i//4)] = prob_string_to_prob('P(NE, A, E)', current_trans_prob)

            elif (i%4) == 1:
                A[i][N+1+len(m_C)+2*(i//4)] = -1 + prob_string_to_prob('P(NE, I, NE)', current_trans_prob)
                A[i][N+2+len(m_C)+2*(i//4)] = prob_string_to_prob('P(NE, I, E)' , current_trans_prob)

            elif (i%4) == 2:
                A[i][N+1+len(m_C)+2*(i//4)] = prob_string_to_prob('P(E, A, NE)', current_trans_prob)
                A[i][N+2+len(m_C)+2*(i//4)] = -1 + prob_string_to_prob('P(E, A, E)', current_trans_prob)

            elif (i%4) == 3:
                A[i][N+1+len(m_C)+2*(i//4)] = prob_string_to_prob('P(E, I, NE)', current_trans_prob)
                A[i][N+2+len(m_C)+2*(i//4)] = -1 + prob_string_to_prob('P(E, I, E)', current_trans_prob)

    b = [0]*(N*4)
    for i in range(N*4):
        if i%4 == 0:
            b[i] = 0
        elif i%4 == 1:
            b[i] = 0
        elif i%4 == 2:
            b[i] = -1
        elif i%4 == 3:
            b[i] = -1

    bound_for_variables = []
    for i in range(3*N+1+len(m_C)):
        if i>=N+1 and i<N+1+len(m_C):
            bound_for_variables.append([0,None])
        else:
            bound_for_variables.append([None,None])



    res = linprog(c, A_ub=A, b_ub=b, bounds=bound_for_variables, method='revised simplex')
    return dict(zip(list(m_C.keys()), res.x[[N+idx+1 for idx in range(len(m_C))]]))


def whittle_LP(k, transition_prob):

    N = transition_prob.shape[0]
   # total number of variables is N + 1 + 2N 
    #we have the following ordering of variables: v1>. . .> v_N> lambda > f10>f11>f20> . . . > fN1 
    c = [1]*N
    c.append(-(N-k))

    for i in range(N*2):
        c.append(0)

#     cluster_no = {}
#     for i in range(len(m_C)):
#         cluster_no[i] = cluster_transition_probabilities[cluster_transition_probabilities['cluster'] == cluster_ids[i]] 

    A = [[0 for i in range(3*N + 1)] for j in range(N*4)]

    for i in range(N*4):
        for j in range(3*N+1):
            if j==(i//4): #the coefficients of v's in the LP
                A[i][j] = -1 
            if j==N and i%2==0:
                A[i][j] = 1 #the coefficient of lambda

            #here transition_prob is a dictionary with N key-value pairs, each key containing the corresponding beneficiaries\
            #transition prob. The beneficiaries in transition_prob are ordered cluster-wise, i.e. beneficiaries from\
            #cluster_id[0] correspond keys 0 to ppl_per_cluster, then beneficiaries from cluster_id[1] correspond\
            #to keys ppl_per_cluster+1 + 2*ppl_per_cluster, and so on.
            #the current_trans_prob contains the present transition prob for the instant of the loop given by i//4.
            current_trans_prob = transition_prob[i//4]

            if (i%4) == 0:
                A[i][N+1+2*(i//4)] = -1 + prob_string_to_prob('P(NE, A, NE)', current_trans_prob)
                A[i][N+2+2*(i//4)] = prob_string_to_prob('P(NE, A, E)', current_trans_prob)

            elif (i%4) == 1:
                A[i][N+1+2*(i//4)] = -1 + prob_string_to_prob('P(NE, I, NE)', current_trans_prob)
                A[i][N+2+2*(i//4)] = prob_string_to_prob('P(NE, I, E)' , current_trans_prob)

            elif (i%4) == 2:
                A[i][N+1+2*(i//4)] = prob_string_to_prob('P(E, A, NE)', current_trans_prob)
                A[i][N+2+2*(i//4)] = -1 + prob_string_to_prob('P(E, A, E)', current_trans_prob)

            elif (i%4) == 3:
                A[i][N+1+2*(i//4)] = prob_string_to_prob('P(E, I, NE)', current_trans_prob)
                A[i][N+2+2*(i//4)] = -1 + prob_string_to_prob('P(E, I, E)', current_trans_prob)

    b = [0]*(N*4)
    for i in range(N*4):
        if i%4 == 0:
            b[i] = 0
        elif i%4 == 1:
            b[i] = 0
        elif i%4 == 2:
            b[i] = -1
        elif i%4 == 3:
            b[i] = -1

    bound_for_variables = []
    for i in range(3*N+1):
        bound_for_variables.append([None,None])



    res = linprog(c, A_ub=A, b_ub=b, bounds=bound_for_variables, method='revised simplex')
    return res.x[N]