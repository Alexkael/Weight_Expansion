import numpy as np
def load(fname):
    z = []
    for epoch in range(1,51):
        w = np.load(fname+'/w_sample_'+str(epoch)+'.npy',allow_pickle=True)[1]
        z.append(w)

    z = np.array(z)

    zz = np.transpose(z, (1,2,4,3,0))
    zz = zz.reshape((zz.shape[0]*zz.shape[1]*zz.shape[2],zz.shape[3],-1))
    return zz

def converiance(x,y):
    x -= np.mean(x)
    y -= np.mean(y)
    z = (np.mean(x*y))/(np.sqrt(np.mean(x*x)*np.mean(y*y)))
    if z==float('Inf') or z==-float('Inf'):
        return 1
    else:
        return z

split = 4    

lr_ = [0.1, 0.03]
drop_rate_ = [0.0, 0.15, 0.3]
width_ = [1., 0.5]
batch_size_ = [64, 256]
weight_decay_ = [0.0, 0.0005]

act = 'relu'
for net in ['vgg11_relu/','vgg16_relu/','vgg19_relu/']:
    sampling_det = []
    for weight_decay in weight_decay_:
        for drop_rate in drop_rate_:
            for width in width_:
                for batch_size in batch_size_:
                    for lr in lr_:
                        fname = net+str(lr)+'_'+str(drop_rate)+'_'+str(width)+'_'+str(batch_size)+'_'+str(weight_decay)+'_'+str(act)   
                        zz = load(fname)    
                        det = []
                        for zzz in zz[:400]:
                            deet = 0
                            for i in range(int(len(zzz)/split)):
                                corr = np.ones((split,split))
                                for x in range(split):
                                    for y in range(split):
                                        if x!=y:
                                            corr[x][y]=converiance(zzz[x+i*split],zzz[y+i*split])
                                deet += np.linalg.det(corr)
                            if not np.isnan(deet):
                                det.append(deet*split/len(zzz))

                        det = np.array(det)    
                        if np.isnan(np.mean(det)):
                            sampling_det.append(0.4)
                        else:
                            sampling_det.append(np.mean(det))
                        print('complete '+fname)
    np.save(net+'sampling_det.npy',np.array(sampling_det))