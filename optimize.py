from constants import *
from sampler import GibbsSampler
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib


def dev_t(t, tu_mean):
    return np.sign(t-tu_mean)*abs(t-tu_mean)**beta


def func(params, *args):
    """
    Computes the value of the objective function required for gradient descent
    """
    
    global counter
    
    counter += 1
    Nums = args[0]
    Numas = args[1]
    Numa = args[2]
    rating_list = args[3]
    t_mean = args[4]
    
    alpha_vu = params[:(U*K)].reshape((U,K), order='F')
    v_u = params[(U*K):2*(U*K)].reshape((U,K), order='F')
    alpha_bu = params[2*(U*K):2*(U*K) + U].reshape((U,1), order='F')
    b_u = params[2*(U*K) + U:2*(U*K) + 2*U].reshape((U,1), order='F')
    alpha_tu = params[2*(U*K) + 2*U:(2*(U*K) + 2*U + U*A)].reshape((U,A), order='F')
    theta_u = params[(2*U*K + 2*U + U*A): (2*U*K + 2*U + 2*U*A)].reshape((U,A), order='F')

    v_m = params[((2*U*K + 2*U + 2*U*A)):((2*U*K + 2*U + 2*U*A) + M*K)].reshape((M,K), order='F')
    b_m = params[((2*U*K + 2*U + 2*U*A)+ M*K):((2*U*K + 2*U + 2*U*A) + M*K + M)].reshape((M,1), order='F')
    theta_m = params[((2*U*K + 2*U + 2*U*A) + M*K + M):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A)].reshape((M,A), order='F')


    M_a = params[((2*U*K + 2*U + 2*U*A) + M*K + M + M*A):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A + A*K)].reshape((A,K), order='F')
    b_o = params[-1]

    num_theta_uma = np.zeros((len(rating_list), A)) 

    for i in range(len(rating_list)): 
        m = rating_list[i]['m']
        u = rating_list[i]['u']
        t = rating_list[i]['t']
        num_theta_uma[i] = np.exp(theta_u[u] + dev_t(t, t_mean[u])*alpha_tu[u] + theta_m[m])
        
    
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum(axis=1)).T

    M_sum = np.dot(num_theta_uma, M_a)
    
    loss1 = 0
    loss3 = 0

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
    
    loss2 = np.multiply(Nums[:,0], np.log(1/(1 + np.exp(-1*(c*r_hat - b))))) + np.multiply(Nums[:,1], np.log(1/(1 + np.exp((c*r_hat - b)))))    
    loss4 = (np.multiply(Numa, np.log(theta_uma))).sum()
    total_loss = loss1.sum() - loss2.sum() - loss3.sum() - loss4.sum() 

    print("Learning Paramater " + str(counter) + "... Loss: " + str(total_loss))
    return total_loss


def fprime(params, *args):

    Nums = args[0]
    Numas = args[1]
    Numa = args[2]
    rating_list = args[3]
    t_mean = args[4]

    alpha_vu = params[:(U*K)].reshape((U,K), order='F')
    v_u = params[(U*K):2*(U*K)].reshape((U,K), order='F')
    
    alpha_bu = params[2*(U*K):2*(U*K) + U].reshape((U,1), order='F')
    b_u = params[2*(U*K) + U:2*(U*K) + 2*U].reshape((U,1), order='F')
    
    alpha_tu = params[2*(U*K) + 2*U:(2*(U*K) + 2*U + U*A)].reshape((U,A), order='F')
    theta_u = params[(2*U*K + 2*U + U*A): (2*U*K + 2*U + 2*U*A)].reshape((U,A), order='F')

    v_m = params[((2*U*K + 2*U + 2*U*A)):((2*U*K + 2*U + 2*U*A) + M*K)].reshape((M,K), order='F')
    b_m = params[((2*U*K + 2*U + 2*U*A)+ M*K):((2*U*K + 2*U + 2*U*A) + M*K + M)].reshape((M,1), order='F')
    theta_m = params[((2*U*K + 2*U + 2*U*A) + M*K + M):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A)].reshape((M,A), order='F')

    M_a = params[((2*U*K + 2*U + 2*U*A) + M*K + M + M*A):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A + A*K)].reshape((A,K), order='F')
    b_o = params[-1]

    num_theta_uma = np.zeros((len(rating_list), A))

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
        gradC_alpha_bu = 0
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

        ############### Final.

        final_grad_vu = np.zeros((U,K))
        final_grad_alpha_vu = np.zeros((U,K))
        
        final_grad_bu = np.zeros((U,1))
        final_grad_alpha_bu = np.zeros((U,1))

        final_grad_thetau = np.zeros((U,A))
        final_grad_alpha_thetau = np.zeros((U,A))

        final_grad_vm = np.zeros((M,K))
        final_grad_b_m = np.zeros((M,1))     
        final_grad_theta_m = np.zeros((M,A))
        
        final_grad_M_a = np.zeros((A,K))
        final_grad_bo = 0.

        final_grad_vu[u] += (rating_error - gradB_factor)*gradA_vu - gradC_vu
        final_grad_alpha_vu[u] += (rating_error - gradB_factor)*gradA_alpha_vu - gradC_alpha_vu
        
        final_grad_bu[u] += (rating_error - gradB_factor)*gradA_bu - gradC_bu
        #print(rating_error.shape , gradB_factor.shape, gradA_alpha_bu.shape , gradC_alpha_bu.shape)
        final_grad_alpha_bu[u] += (rating_error - gradB_factor)*gradA_alpha_bu - gradC_alpha_bu

        final_grad_thetau[u] += (rating_error - gradB_factor)*gradA_thetau - gradD_thetau
        final_grad_alpha_thetau[u] += (rating_error - gradB_factor)*gradA_alpha_thetau - gradD_alpha_thetau

        final_grad_vm[m] += (rating_error - gradB_factor)*gradA_vm - gradC_vm
        final_grad_b_m[m] += (rating_error - gradB_factor)*gradA_b_m - gradC_b_m      
        final_grad_theta_m[m] += (rating_error - gradB_factor)*gradA_theta_m - gradD_thetam
        
        final_grad_M_a += (rating_error - gradB_factor)*gradA_M_a - gradC_M_a
        
        final_grad_bo += (rating_error - gradB_factor)*gradA_bo - gradC_bo

        return numpy.concatenate((final_grad_alpha_vu.flatten('F'), 
            final_grad_vu.flatten('F'), 
            final_grad_alpha_bu.flatten('F'), 
            final_grad_bu.flatten('F'), 
            final_grad_alpha_thetau.flatten('F'), 
            final_grad_thetau.flatten('F'), 
            final_grad_vm.flatten('F'), 
            final_grad_b_m.flatten('F'), 
            final_grad_theta_m.flatten('F'), 
            final_grad_M_a.flatten('F'),
            np.array([final_grad_bo]).flatten('F')))


# def fprime(params, *args):
#     return np.ones((len(params)))


def optimizer(Nums,Numas,Numa,rating_list,t_mean):
    """
    Computes the optimal values for the parameters required by the JMARS model using lbfgs
    """
    global counter

    args = (Nums,Numas,Numa,rating_list,t_mean)
    initial_values = numpy.concatenate((alpha_vu.flatten('F'), 
                                        v_u.flatten('F'), 
                                        alpha_bu.flatten('F'), 
                                        b_u.flatten('F'), 
                                        alpha_tu.flatten('F'), 
                                        theta_u.flatten('F'), 
                                        v_m.flatten('F'), 
                                        b_m.flatten('F'), 
                                        theta_m.flatten('F'), 
                                        M_a.flatten('F'), 
                                        np.array([b_o]).flatten('F')))
    
    x,f,d = fmin_l_bfgs_b(func, x0=initial_values, fprime=fprime, args=args, approx_grad=False, maxfun=1, maxiter=10)
    counter = 0

    return x,f,d