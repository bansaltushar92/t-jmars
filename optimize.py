from constants import *
from sampler import GibbsSampler
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib


# define t_mean, beta?

def dev_t(t, tu_mean):
    return np.sign(t-tu_mean)*abs(t-tu_mean)**beta


def func(params, *args):
    """
    Computes the value of the objective function required for gradient descent
    """
    global counter
    
    counter += 1
    #y = args[0]
    #z = args[1]
    #s = args[2]
    Nums = args[0]
    Numas = args[1]
    Numa = args[2]
    rating_list = args[3]
    t_mean = args[4]
    
    #v_ut = np.zeros((R,K))
    alpha_vu = params[:(U*K)].reshape((U,K), order='F')
    v_u = params[(U*K):2*(U*K)].reshape((U,K), order='F') #+ dev_t(t)*alpha_vu
    alpha_bu = params[2*(U*K):2*(U*K) + U].reshape((U,1), order='F')                # change indices all after this.
    b_u = params[2*(U*K) + U:2*(U*K) + 2*U].reshape((U,1), order='F') #+ dev_t(t)*alpha_bu
    alpha_tu = params[2*(U*K) + 2*U:(2*(U*K) + 2*U + U*A)].reshape((U,A), order='F')
    theta_u = params[(2*U*K + 2*U + U*A): (2*U*K + 2*U + 2*U*A)].reshape((U,A), order='F') #+ dev_t(t)*alpha_tu

    v_m = params[((2*U*K + 2*U + 2*U*A)):((2*U*K + 2*U + 2*U*A) + M*K)].reshape((M,K), order='F')
    b_m = params[((2*U*K + 2*U + 2*U*A)+ M*K):((2*U*K + 2*U + 2*U*A) + M*K + M)].reshape((M,1), order='F')
    theta_m = params[((2*U*K + 2*U + 2*U*A) + M*K + M):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A)].reshape((M,A), order='F')

    # num_theta_uma = np.exp(np.tile(theta_ut.reshape(U,1,A), (1,M,1)) + np.tile(theta_m.reshape(1,M,A), (U,1,1)))
    # theta_uma = num_theta_uma / (num_theta_uma.sum())
    M_a = params[((2*U*K + 2*U + 2*U*A) + M*K + M + M*A):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A + A*K)].reshape((A,K), order='F')

    # M_a_norm = np.dot(num_theta_uma, M_a)

    # M_sum = np.diag(M_a_norm.sum(0))

    # print (v_ut.shape, M_a_norm.shape, M_sum.shape, v_m.shape, b_ut.shape, b_m.shape)
    # r_hat =  np.dot(np.dot(v_ut, M_sum), v_m.T) + b_o*np.ones((U,M)) + np.matlib.repmat(b_ut,1,M) + np.matlib.repmat(b_m.T,U,1)


    num_theta_uma = np.zeros((len(rating_list), A)) # How to define
    
    # Make something in indexer which stores only user-movie pairs for given data.

    for i in range(len(rating_list)): #Get ratinglist
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        #print(np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m]))
        num_theta_uma[i] = np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m])
        
    
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum(axis=1)).T

    M_sum = np.dot(num_theta_uma, M_a)
    
    loss1 = 0
    loss3 = 0
    #print("2")
    for i in range(len(rating_list)):
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        r = rating_list[i]['r']

        v_ut = v_u[u] + dev_t(t, t_mean[u])*alpha_vu[u]
        b_ut = b_u[u] + dev_t(t, t_mean[u])*alpha_bu[u]
        r_hat[i] =  np.dot(np.dot(v_ut, np.diag(M_sum[i])), v_m[m].T) + b_o + b_ut + b_m[m]
        loss1 += epsilon*(r - r_hat[i])
        
        for j in range(A):
            ruma = np.dot(np.dot(v_ut, np.diag(M_a[j])), v_m[m].T) + b_o + b_ut + b_m[m]
            # sentiment - 0 -> positive,  1-> negative
            loss3 += np.multiply(Numas[i,j,0], np.log(1/(1 + np.exp(-1*(c*ruma - b))))) + np.multiply(Numas[i,j,1], np.log(1/(1 + np.exp((c*ruma - b)))))

    # r_hat[u][m] =  np.dot(np.dot(v_u[u] + dev_t(t)*alpha_vu[u], M_sum), v_m[m].T) + b_o + (b_u[u] + dev_t(t)*alpha_bu[u]) + b_m[m]

    
    loss2 = np.multiply(Nums[:,0], np.log(1/(1 + np.exp(-1*(c*r_hat - b))))) + np.multiply(Nums[:,1], np.log(1/(1 + np.exp((c*r_hat - b)))))
    
    #loss3 = np.zeros((U,M))
    
    loss4 = (np.multiply(Numa, np.log(theta_uma))).sum()

    total_loss = loss1.sum() - loss2.sum() - loss3.sum() - loss4.sum()
    #loss = np.multiply(loss, (rating_matrix > 0))
    
    print("Learning Paramater " + str(counter) + "... Loss: " + str(total_loss))
    return total_loss


def fprime(params, *args):

    ## pass b_o through parameters and find it's gradient

    Nums = args[0]
    Numas = args[1]
    Numa = args[2]
    rating_list = args[3]
    t_mean = args[4]

    num_theta_uma = np.zeros((len(rating_list), A)) # How to define

    for i in range(len(rating_list)):
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']

        num_theta_uma[i] = np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m])
        
    
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum(axis=1)).T
    M_sum = np.dot(num_theta_uma, M_a)

    for i in range(len(rating_list)):
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t'] 
        r = rating_list[i]['r']
        
        v_ut = v_u[u] + dev_t(t, t_mean[u])*alpha_vu[u]
        b_ut = b_u[u] + dev_t(t, t_mean[u])*alpha_bu[u]
        r_hat = np.dot(np.dot(v_ut, np.diag(M_sum[i])), v_m[m].T) + b_o + b_ut + b_m[m]
        rating_error = 2*epsilon*(r_hat - r)
        
        ########## For term A
        gradA_vu = np.multiply(np.dot(theta_uma[i], M_a), v_m[m].T)
        gradA_alpha_vu = gradA_vu*dev_t(t, t_mean[u])
        
        #grad_vu[u] += rating_error * grad_vu_temp

        gradA_bu = 1
        gradA_alpha_bu = dev_t(t, t_mean[u])

        gradA_theta = np.multiply( np.dot( np.multiply(v_ut, M_a), v_m[m]), (theta_uma[i] * (1 - theta_uma[i]))) 
        gradA_thetau = gradA_theta
        gradA_alpha_thetau = gradA_thetau*dev_t(t, t_mean[u])

        gradA_vm = np.multiply(v_ut, np.dot(theta_uma[i], M_a))
        gradA_b_m = 1        
        gradA_theta_m = gradA_theta
        
        gradA_M_a = np.dot(theta_uma[i].T, np.multiply(v_ut, v_m[m]))
        gradA_bo = 1 

        
        ########### For term B
        gradB_factor =  (Nums[i,0] /(1 + np.exp(c*r_hat - b)))*1*c + (Nums[i,1]/(1 + np.exp(-1*(c*r_hat - b))))*(-1)*c

        ########### For term C
        gradC_vu = np.zeros(K)
        gradC_alpha_vu = np.zeros(K)
        gradC_bu = 0 
        gradC_alpha_bu = np.zeros(K)
        gradC_vm = np.zeros(K)
        gradC_b_m = 0
        gradC_M_a = np.zeros(K)
        gradC_bo = 0

        for j in range(A):
            ruma = np.dot(np.dot(v_ut, np.diag(M_a[j])), v_m[m].T) + b_o + b_ut + b_m[m]
            gradC_factor = (Numas[i,j,0]/(1 + np.exp(c*ruma - b)))*1*c + (Numas[i,j,1]/(1 + np.exp(-1*(c*ruma - b))))*(-1)*c

            gradC_vu += gradC_factor*np.multiply(M_a[j], v_m[m])
            gradC_alpha_vu += gradC_factor*gradC_vu*dev_t(t, t_mean[u])
            
            gradC_bu += gradC_factor
            gradC_alpha_bu += gradC_factor*dev_t(t, t_mean[u])

            gradC_vm += gradC_factor*np.multiply(M_a[j], v_ut)
            gradC_b_m += gradC_factor        
            
            gradC_M_a += gradC_factor*np.multiply(v_ut, v_m[m])
            gradC_bo += gradC_factor

        ########### For term D
        
        softmax_matrix = np.zeros((A,A))
        numa_by_theta_uma = np.zeros(A)

        for j in range(A):
            numa_by_theta_uma[j] = Numa[i,j]/theta_uma[i][j]
            for k in range(A):
                if j == k:
                    softmax_matrix[j][k] = theta_uma[i][j]*(1-theta_uma[i][k])
                else:
                    softmax_matrix[j][k] = -1*theta_uma[i][j]*theta_uma[i][k]

        gradD_thetau = np.dot(numa_by_theta_uma, softmax_matrix)
        gradD_thetam = np.dot(numa_by_theta_uma, softmax_matrix)

        gradD_alpha_thetau = gradD_thetau*dev_t(t, t_mean[u])




        ############### Final. ->>>>> Define all.

        final_grad_vu[u] += (rating_error - gradB_factor)*gradA_vu - gradC_vu
        final_grad_alpha_vu[u] += (rating_error - gradB_factor)*gradA_alpha_vu - gradC_alpha_vu
        
        final_grad_bu[u] += (rating_error - gradB_factor)*gradA_bu - gradC_bu
        final_grad_alpha_bu[u] += (rating_error - gradB_factor)*gradA_alpha_bu - gradC_alpha_bu

        final_grad_thetau[u] += (rating_error - gradB_factor)*gradA_thetau - gradD_thetau
        final_grad_alpha_thetau[u] += (rating_error - gradB_factor)*gradA_alpha_thetau - gradD_alpha_thetau

        final_grad_vm[m] += (rating_error - gradB_factor)*gradA_vm - gradC_vm
        final_grad_b_m[m] += (rating_error - gradB_factor)*gradA_b_m - gradC_b_m      
        final_grad_theta_m[m] += (rating_error - gradB_factor)*gradA_theta_m - gradD_thetam
        
        final_grad_M_a += (rating_error - gradB_factor)*gradA_M_a - gradC_M_a
        final_grad_bo += (rating_error - gradB_factor)*gradA_bo - gradC_bo

    # complete all gradients.

    # gradB_vut = np.dot(np.dot(theta_ut, M_a), v_m.T)
    # gradB_vu = grad_vut
    # gradB_alpha_vu = grad_vut*dev_t(t, t_mean)

    # gradB_bu = 1
    # gradB_alpha_bu = dev_t(t, t_mean)

    # gradB_thetaut = np.multiply( np.dot( np.multiply(v_ut.T, M_a), v_m), num_theta_uma) 
    # gradB_thetau = grad_thetaut
    # gradB_alpha_thetau = grad_thetaut*dev_t(t, t_mean)

    # gradB_v_m = np.dot(vu, np.dot(theta_ut, M_a))
    # gradB_b_m = np.ones((U,1))
    # gradB_theta_m = np.multiply( np.dot( np.multiply(v_ut.T, M_a), v_m), num_theta_uma)
    # gradB_M_a = np.dot(num_theta_uma, np.dot(v_ut, v_m.T).reshape((1, K, K)))
    
    # gradC_factor = 0

    # for i in range(A):
    #     gradC_factor += np.multiply(Numas[:,:,i,0], 1/(1 + np.exp((c*ruma - b))))*1*c + np.multiply(Numas[:,:,i,1], 1/(1 + np.exp(-1*(c*ruma - b))))*(-1)*c
    
    # gradD_thetau = np.multiply(np.multiply(np.divide(Numa, theta_uma), theta_uma), np.ones(A) - theta_uma)
    # gradD_thetam = gradD_thetau






'''def fprime(params, *args):
    y = args[0]
    z = args[1]
    s = args[2]
    Nums = args[3]
    Numas = args[4]
    Numa = args[5]

    v_u = params[:(U*K)].reshape((U,K), order='F')
    b_u = params[(U*K):(U*K + U)].reshape((U,1), order='F')
    theta_u = params[(U*K + U):(U*K + U + U*A)].reshape((U,A), order='F')

    v_m = params[(U*K + U + U*A):(U*K + U + U*A + M*K)].reshape((M,K), order='F')
    b_m = params[(U*K + U + U*A + M*K):(U*K + U + U*A + M*K + M)].reshape((M,1), order='F')
    theta_m = params[(U*K + U + U*A + M*K + M):(U*K + U + U*A + M*K + M + M*A)].reshape((M,A), order='F')

    M_a = params[(U*K + U + U*A + M*K + M + M*A):].reshape((A,K), order='F')

    M_sum = np.diag(M_a.sum(0))

    grad_vu = np.zeros((v_u.shape))
    grad_bu = np.zeros((b_u.shape))
    grad_thetau = np.zeros((theta_u.shape))

    grad_vm = np.zeros((v_m.shape))
    grad_bm = np.zeros((b_m.shape))
    grad_thetam = np.zeros((theta_m.shape))

    grad_Ma = np.zeros((M_a.shape))

    r_hat =  np.dot(np.dot(v_u, M_sum), v_m.T) + b_o*np.ones((U,M)) + np.matlib.repmat(b_u,1,M) + np.matlib.repmat(b_m.T,U,1)
    theta_uma = np.exp(np.tile(theta_u.reshape(U,1,A), (1,M,1)) + np.tile(theta_u.reshape(1,M,A), (U,1,1)))
    theta_uma = theta_uma / (theta_uma.sum())

    for u in range(U):
        for m in range(M):
            if rating_matrix[u][m] != 0:
                ruma = 


    #partial derivatives of ruma
    grad_ruma_vu = np.zeros((U,M,A,K))
    grad_ruma_vm = np.zeros((M,U,A,K))
    for i in range(A):
        grad_ruma_vu[:,:,i,:] = np.tile(np.multiply(np.matlib.repmat(M_a[i],M,1), v_m).reshape(M,1,K), (U,1,1,1))
        grad_ruma_vm[:,:,i,:] = np.tile(np.multiply(np.matlib.repmat(M_a[i],U,1), v_u).reshape(U,1,K), (M,1,1,1))

    grad_ruma_bu = np.ones((U,M,A,K))
    grad_ruma_bm = np.ones((U,M,A,K))
    grad_ruma_thetaua = np.zeros((U,M,A,K))
    grad_ruma_thetama = np.zeros((U,M,A,K))

    for i in range(K):
        grad_ruma_mak[:,:,:,i] =  v_u[:]  np.tile(np.multiply(np.matlib.repmat(M_a[i],U,1), v_u).reshape(U,1,K), (M,1,1,1))

    grad_vu = -2*epsilon*np.dot(np.multiply((rating_matrix - r_hat), (rating_matrix > 0)), np.dot(theta_uma, M_a))'''

def fprime(params, *args):
    return np.ones((len(params)))


def optimizer(Nums,Numas,Numa,rating_list,t_mean):
    """
    Computes the optimal values for the parameters required by the JMARS model using lbfgs
    """
    global counter

    #params = [v_u, b_u, theta_u, v_m, b_m, theta_m, M_a]
    #initial_values = np.array([v_u, b_u, theta_u, v_m, b_m, theta_m, M_a], dtype=object)
    #print func(initial_values, *args)


    args = (Nums,Numas,Numa,rating_list,t_mean)
    #print(len(rating_list))

    initial_values = numpy.concatenate((alpha_vu.flatten('F'), v_u.flatten('F'), alpha_bu.flatten('F'), b_u.flatten('F'), alpha_tu.flatten('F'), theta_u.flatten('F'), v_m.flatten('F'), b_m.flatten('F'), theta_m.flatten('F'), M_a.flatten('F')))    

    x,f,d = fmin_l_bfgs_b(func, x0=initial_values, fprime=fprime, args=args, approx_grad=False, maxfun=1, maxiter=1)
    counter = 0

    #print x
    #print f
    #print d

    return x,f,d