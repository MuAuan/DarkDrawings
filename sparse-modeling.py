"""
uk+1=minimize_u |dx|+|dy|+\frac{λ}{2}|u−I|^2+\frac{μ}{2}(|∇_xu−dx−b^k_x|^2+|∇_yu−dy−b^k_y|^2)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
   
def add_noise(I_t):
    mu = np.mean(I_t)
    sigma = np.std(I_t)
    dB = 3
    I_noise = 10**(-dB/20)*np.reshape([random.gauss(mu, sigma) for i in range(np.size(I_t))], np.shape(I_t))
    print(np.max(I_noise), np.min(I_noise))
    I = I_t + I_noise
    max_I  = np.max(I)
    min_I = np.min(I)
    I = (I - min_I)/(max_I - min_I)
    return I

"""
u_{k+1}=minimize_u |d_x|+|d_y|+\frac{λ}{2}|u−I|^2+\frac{μ}{2}(|∇_xu−dx−b^k_x|^2+|∇_yu−dy−b^k_y|^2)
"""
def Gauss_Saidel(u, d_x, d_y, b_x, b_y, f, MU, LAMBDA,X_N,Y_N):
    U = np.hstack([u[:,1:X_N], np.reshape(u[-1,:],[Y_N,1] )]) + np.hstack([np.reshape(u[0,:],[Y_N,1]), u[:,0:Y_N-1]]) \
       + np.vstack([u[1:X_N,:], np.reshape(u[:,-1],[1,X_N] )]) + np.vstack([np.reshape(u[:,0],[1,X_N] ), u[0:X_N-1,:]])
    D = np.vstack([np.reshape(d_x[:,0],[1,X_N] ), d_x[0:Y_N-1,:]]) - d_x \
       + np.hstack([np.reshape(d_y[0,:],[Y_N,1] ), d_y[:,0:X_N-1]]) - d_y
    B = -np.vstack([np.reshape(b_x[:,0],[1,X_N] ), b_x[0:Y_N-1,:]]) + b_x \
       - np.hstack([np.reshape(b_y[0,:],[Y_N,1] ), b_y[:,0:X_N-1]]) + b_y
    G = LAMBDA/(MU + 4*LAMBDA)*(U+D+B) + MU/(MU + 4*LAMBDA)*f
    return G
    

def shrink(x,y):
    t = np.abs(x) - y
    S = np.sign(x)*(t > 0) * t
    return S

def cycle_Gauss_Seidel(f,X_N,Y_N):
    CYCLE = 10000 #1000 #300 #200 #100
    MU = 0.05 #5 #5.0*10**(-2)
    LAMBDA = 0.01 #0.1 #1 #1.0*10**(-2)
    TOL = 1e-2 #3 #-1 #-5 #5.0*10**(-1)
    X_N,Y_N=X_N,Y_N

    ## Initialization
    u = f
    d_x = np.zeros([X_N,Y_N])
    d_y = np.zeros([X_N,Y_N])
    b_x = np.zeros([X_N,Y_N])
    b_y = np.zeros([X_N,Y_N])

    for cyc in range(CYCLE):
        u_n = Gauss_Saidel(u,d_x,d_y, b_x ,b_y,f, MU,LAMBDA,X_N,Y_N)
        Err = np.max(np.abs(u_n[2:X_N-2,2:Y_N-2] - u[2:X_N-2,2:Y_N-2]))
        if np.mod(cyc,200)==0:
            print([cyc,Err])
        if Err < TOL:
            break
        else:
            u = u_n
            nablax_u = np.vstack([u[1:X_N,:], np.reshape(u[:,-1],[1,X_N] )]) - u 
            nablay_u = np.hstack([u[:,1:X_N], np.reshape(u[-1,:],[Y_N,1] )]) - u 
            d_x = shrink(nablax_u + b_x, 1/LAMBDA)
            d_y = shrink(nablay_u + b_y, 1/LAMBDA)
            b_x = b_x + (nablax_u - d_x)
            b_y = b_y + (nablay_u - d_y)
    return u

def plot_fig(img='img_load',fig_name='default'):
    plt.imshow(img)
    plt.title(fig_name)
    plt.savefig('Original_'+str(fig_name)+'.jpg')
    plt.pause(1)
    plt.close()

def main():
    img_rows, img_cols = 400,400
    img_path = "Blood2.jpg" #"s7.jpg" #"Blood2.jpg"
    img_load = cv2.imread(img_path)
    img_load = cv2.resize(img_load, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    img_load = img_load[:,:,::-1]

    plot_fig(img_load,'input')
    img_load_o=img_load
    r_o, g_o, b_o = cv2.split(img_load_o)
    r_no=(r_o-np.min(r_o)) /(np.max(r_o)-np.min(r_o))
    g_no=(g_o-np.min(g_o)) /(np.max(g_o)-np.min(g_o))
    b_no=(b_o-np.min(b_o)) /(np.max(b_o)-np.min(b_o))
    
    rgb_str= img_path+'rgb_2'  #'YCrCb_2' #'YCrCb_5' #'rgb_5'  #'rgb_2' #'LAB'  #'YUV'  #'HSV' #'YCR_CB' #'rgb'
    #img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2YCrCb)
    #img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2HSV)
    #img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2YUV)
    #img_load = cv2.cvtColor(img_load, cv2.COLOR_BGR2LAB)
    r, g, b = cv2.split(img_load) #cv2.cvtColor(img_load, cv2.COLOR_RGB2GRAY)
    r_nco=(r-np.min(r)) /(np.max(r)-np.min(r))
    g_nco=(g-np.min(g)) /(np.max(g)-np.min(g))
    b_nco=(b-np.min(b)) /(np.max(b)-np.min(b))
    
    plt.figure()
    plt.subplot(321)
    plt.axis("off")
    plt.imshow(img_load_o)
    plt.title("Original")

    img_load_on = cv2.merge((r_no,g_no,b_no))
    plt.subplot(322)
    plt.axis("off")
    plt.imshow(img_load_on)
    plt.title("Original_norm")
    
    
    plt.subplot(323)
    plt.axis("off")
    plt.imshow(img_load)
    plt.title("Converted")

    img_load_ic = cv2.merge((r_nco,g_nco,b_nco))
    plt.subplot(324)
    plt.axis("off")
    plt.imshow(img_load_ic)
    plt.title("Convd_norm")
    
    #img_load_c = cv2.cvtColor(img_load, cv2.COLOR_YCrCb2BGR)
    #img_load_c = cv2.cvtColor(img_load, cv2.COLOR_LAB2BGR)
    #img_load_c = cv2.cvtColor(img_load, cv2.COLOR_YUV2BGR)
    #img_load_c = cv2.cvtColor(img_load, cv2.COLOR_HSV2BGR)
    plt.subplot(325)
    plt.axis("off")
    #plt.imshow(img_load_c)
    plt.title("Inversed")
    
    #img_load_ic = np.array(img_load_ic, dtype=np.float32) #np.float32)  #np.uint8)
    #img_load_ic = cv2.cvtColor(np.clip(img_load_ic, 0, 1), cv2.COLOR_YCrCb2BGR)
    #img_load_ic = cv2.cvtColor(img_load_ic, cv2.COLOR_LAB2BGR)
    #img_load_ic = cv2.cvtColor(img_load_ic, cv2.COLOR_YUV2BGR)
    #img_load_ic = cv2.cvtColor(img_load_ic, cv2.COLOR_HSV2BGR)
    plt.subplot(326)
    plt.axis("off")
    plt.imshow(np.clip(img_load_ic, 0, 1))
    plt.title("Inverse_norm")
    plt.savefig('Original_Converted_normalized_'+rgb_str+'_.jpg')

    plt.pause(1)
    plt.close()
    
    
    list=(r,g,b)

    s=0
    rgb=[]
    rgbs=[]
    for I_t in list:
        plot_fig(I_t,'Original'+rgb_str+str(s))
        
        f = add_noise(I_t) #I_t #add_noise(I_t)
        plot_fig(f,'Noise_'+rgb_str+'_'+str(s))
        
        [X_N,Y_N] = np.shape(f)
        print(X_N,Y_N)

        u = cycle_Gauss_Seidel(f,X_N,Y_N)
        #u=f
        ## plot figure
        plt.figure()
        
        plt.subplot(1,3,1)
        plt.axis("off")
        plt.imshow(I_t)
        plt.title("Original")
                
        plt.subplot(1,3,2)
        plt.axis("off")
        plt.imshow(f)
        plt.title("Noisy")
        
        plt.subplot(1,3,3)
        plt.axis("off")
        plt.imshow(u)
        #x1, y1 = [0,X_N], [X_N,Y_N]  #[50,50]
        #plt.plot(x1, y1)
        plt.title('Reconstructed')
        plt.savefig('Reconstructed_'+rgb_str+'_'+str(s)+'.jpg')
        plt.close()
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(f[50,:])
        plt.subplot(2,1,2)
        plt.plot(u[50,:])
        plt.savefig('Tracing_'+rgb_str+'_'+str(s)+'.jpg')
        plt.pause(1)
        plt.close()
        if s==0:
            r=(u-np.min(u)) /(np.max(u)-np.min(u))  #/255
            f_0=f
        elif s==1:
            g=(u-np.min(u)) /(np.max(u)-np.min(u))  #/255
            f_1=f
        else:
            b=(u-np.min(u)) /(np.max(u)-np.min(u))  #/255
            f_2=f
        s+=1
    #plt.close()    
    rgbs = cv2.merge((r,g,b))
    plt.imshow(rgbs)
    plt.savefig('Reconstructed_total_'+rgb_str+'.jpg')
    plt.pause(1)
    plt.close()
    
    fs = cv2.merge((f_0,f_1,f_2))
    fs = np.array(fs, dtype=np.float32)  #np.uint8)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_YCrCb2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_LAB2BGR)
    f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_YUV2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_HSV2BGR)
    #plt.axis("off")
    plt.imshow(np.clip(f_inv, 0, 1))
    plt.title("Inverse_Noisy")    
    plt.savefig('Inv_Noisy_'+rgb_str+'_.jpg')
    plt.pause(1)
    plt.close()
 
    
    rgbs1 = np.array(rgbs, dtype=np.float32)  #np.uint8)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_YCrCb2BGR)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_LAB2BGR)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_YUV2BGR)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_HSV2BGR)
    #plt.axis("off")
    plt.imshow(np.clip(rgbs1, 0, 1))
    plt.title("Inverse_Recon")
    plt.savefig('Inv_Reconstructed_'+rgb_str+'_.jpg')
    plt.pause(1)
    plt.close()
    

    plt.figure()
    fs = cv2.merge((f_0,f_1,f_2))
    fs = np.array(fs, dtype=np.float32)  #np.uint8)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_YCrCb2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_LAB2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_YUV2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_HSV2BGR)
    plt.subplot(121)
    #plt.axis("off")
    plt.imshow(np.clip(f_inv, 0, 1))
    plt.title("Inverse_Noisy")    
    
    rgbs1 = np.array(rgbs, dtype=np.float32)  #np.uint8)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_YCrCb2BGR)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_LAB2BGR)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_YUV2BGR)
    #rgbs1 = cv2.cvtColor(np.clip(rgbs1, 0, 1), cv2.COLOR_HSV2BGR)
    plt.subplot(122)
    #plt.axis("off")
    plt.imshow(np.clip(rgbs1, 0, 1))
    plt.title("Inverse_Recon")
    plt.savefig('Inv_Reconst_'+rgb_str+'_.jpg')
    plt.pause(1)
    plt.close()
    
    plt.figure()
    plt.subplot(341)
    plt.axis("off")
    plt.imshow(img_load_o)
    plt.title("Original")
    
    plt.subplot(342)
    plt.axis("off")
    plt.imshow(img_load)
    plt.title("Converted")

    img_load_ic = cv2.merge((r_nco,g_nco,b_nco))
    plt.subplot(343)
    plt.axis("off")
    plt.imshow(img_load_ic)
    plt.title("Convd_norm")
    
    rgb_o=cv2.merge((r_no,g_no,b_no))
    plt.subplot(344)
    plt.axis("off")
    plt.imshow(rgb_o)
    plt.title("Normalized")
    
    fs = cv2.merge((f_0,f_1,f_2))
    plt.subplot(347)
    plt.axis("off")
    plt.imshow(f)
    plt.title("Noisy_norm")

    fs = np.array(fs, dtype=np.float32)  #np.uint8)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_YCrCb2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_LAB2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_YUV2BGR)
    #f_inv = cv2.cvtColor(np.clip(fs, 0, 1), cv2.COLOR_HSV2BGR)
    plt.subplot(348)
    plt.axis("off")
    plt.imshow(np.clip(f_inv, 0, 1))
    plt.title("Inverse_Noisy")
    
    plt.subplot(3,4,11)
    plt.axis("off")
    plt.imshow(rgbs)
    plt.title('Reconstructed')
    
    rgbs = np.array(rgbs, dtype=np.float32)  #np.uint8)
    #rgbs = cv2.cvtColor(np.clip(rgbs, 0, 1), cv2.COLOR_YCrCb2BGR)
    #rgbs = cv2.cvtColor(np.clip(rgbs, 0, 1), cv2.COLOR_LAB2BGR)
    #rgbs = cv2.cvtColor(np.clip(rgbs, 0, 1), cv2.COLOR_YUV2BGR)
    #rgbs = cv2.cvtColor(np.clip(rgbs, 0, 1), cv2.COLOR_HSV2BGR)
    plt.subplot(3,4,12)
    plt.axis("off")
    plt.imshow(np.clip(rgbs, 0, 1))
    plt.title("Inverse_Recon")
    plt.savefig('Original_Noisy_Reconstructed_'+rgb_str+'_.jpg')
    plt.pause(1)
    plt.close()

if __name__ == "__main__":
    main()    