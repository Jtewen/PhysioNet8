import numpy as np
from scipy.signal import filtfilt, butter
from scipy.io import loadmat
import os



def to_one_hot(array, N):
	array = np.array(array).astype(int)
	labels = np.zeros((array.shape[0], N), dtype=np.float32)
	labels[np.arange(array.shape[0]), array] = 1.
	return labels



def calc_domain_mask(domains, labels):
    
    ref_domains = list(set(domains))
    masks = np.zeros_like(labels)
    for d in ref_domains:
        inds = np.argwhere(domains == d).ravel()
        domain_labels = labels[inds]
        domain_mask = domain_labels.sum(axis=0)
        domain_mask[domain_mask > 0] = 1
        masks[inds] = domain_mask
        
    return masks



def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')



def calc_weights(labels, weights, identity_w=0.75):
    new_weights = 1 - weights + identity_w * np.eye(len(weights))
    label_weights = []
    for i in range(len(labels)):
        label = labels[i]
        one_inds = np.argwhere(label).ravel()
        weight = new_weights[one_inds].mean(0)
        label_weights.append(weight)
    
    return np.array(label_weights, dtype='float32')


def find_domains(names):
    domains = []
    for i in range(len(names)):
        domains.append(names[i][:1])
    ref_domains = list(set(domains))
    
    domains = np.array(domains)
    
    domains_int = np.ones_like(domains, dtype='uint8')
    for i, ref_domain in enumerate(ref_domains):
        inds = np.argwhere(domains == ref_domain)
        domains_int[inds] = i
        
    return domains_int


def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header



def resample(data, src_frq, trg_frq=500):
    N_src = data.shape[0]
    N_trg = int(N_src * trg_frq / src_frq)
    
    resampled = np.zeros((N_trg, data.shape[1]), dtype='float32')
    for i in range(data.shape[1]):
        resampled[:,i] = np.interp(np.linspace(0, N_src, N_trg), np.arange(N_src), data[:, i])
        
    return resampled


def butter_lowpass(cutoff, fs=125, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs=125, order=6):
    b, a = butter_lowpass(cutoff, fs=fs, order=order)
    y = filtfilt(b, a, data)
    return y


def find_st_end(data_len, final_len):
    
    st = 0
    end = data_len
    
    great = max(data_len, final_len)
    small = min(data_len, final_len)
    
    if great != small:
        diff = great - small
        
        st = np.random.randint(0, diff + 1)
        end = st + small
    
    return st, end

def prepare_data(x, length, mod=0, scale=1000., clip1=-100., clip2=100., aug=True,
        p_art1=0.025, p_art2=0.02, p_noise1=0.02, p_noise2=0.02, p_filt1=0.03, p_filt2=0.025,
        p_shfl=0.02, p_chzero1=0.025, p_chzero2=0.02, p_inv1=0.02, p_inv2=0.02, p_inv3=0.02,
        p_scale=0.1, p_scale2=0.03, p_shft=0.06):

    data = np.zeros((len(x), length, 12))
    for i in range(len(x)):
        sig = x[i][mod::2].copy()
        L = sig.shape[0]
        st, end = find_st_end(L, length)
        if L > length:
            data[i] = sig[st: end]
        else:
            data[i, st: end] = sig
            
    data = np.clip(data / scale, clip1, clip2)
    
    if not aug:
        return data
    
    for i in range(len(data)):

        if np.random.rand() < p_chzero1:
            ind = np.random.randint(12)
            data[i,:,ind] *= 0.
            if np.random.rand() < 0.5:
                fc = np.random.randint(2, 5)
                n_ratio = (0.1 + np.random.rand()) * 0.5
                data[i,:,ind] = lowpass_filter(np.random.normal(0, n_ratio, size=data[i,:,ind].shape), fc)

        elif np.random.rand() < p_chzero2:
            inds = np.random.choice(np.arange(12), size=np.random.randint(2,6)).ravel()
            data[i,:,inds] *= 0.
            
            for ind in inds:
                if np.random.rand() < 0.3:
                    fc = np.random.randint(2, 5)
                    n_ratio = (0.1 + np.random.rand()) * 0.5
                    data[i,:,ind] = lowpass_filter(np.random.normal(0, n_ratio, size=data[i,:,ind].shape), fc)
            
        if np.random.rand() < p_noise1:
            n_ratio = np.random.rand() * 0.03 + 0.001
            inds = np.random.choice(np.arange(12), size=np.random.randint(12), replace=False).ravel()
            data[i,:,inds] += np.random.normal(0, n_ratio, size=data[i,:,inds].shape)
            
        elif np.random.rand() < p_noise2:
            n_ratio = np.random.rand() * 0.03 + 0.001
            inds = np.random.choice(np.arange(12), size=np.random.randint(12), replace=False).ravel()
            
            t_inds = np.sort(np.random.choice(np.arange(length), size=10, replace=False).ravel())
            
            ai = np.random.randint(6)
            a = t_inds[ai]
            b = t_inds[np.random.randint(ai+2, 10)]
            data[i,a:b,inds] += np.random.normal(0, n_ratio, size=data[i,a:b,inds].shape)

        try:
            if np.random.rand() < p_art1:
                
                inds = np.random.choice(np.arange(12), size=np.random.randint(12), replace=False).ravel()
                for ind in inds:
                    fc = np.random.randint(2, 5)
                    n_ratio = (0.1 + np.random.rand()) * 0.2
                    data[i,:,ind] += lowpass_filter(np.random.normal(0, n_ratio, size=data[i,:,ind].shape), fc)
                
            elif np.random.rand() < p_art2:
                
                inds = np.random.choice(np.arange(12), size=np.random.randint(12), replace=False).ravel()
                for ind in inds:
                    fc = np.random.randint(2, 5)
                    n_ratio = (0.1 + np.random.rand()) * 0.2
                    t_inds = np.sort(np.random.choice(np.arange(length), size=10, replace=False).ravel())
                    ai = np.random.randint(6)
                    a = t_inds[ai]
                    b = t_inds[np.random.randint(ai+3, 10)]
                    data[i,a:b,ind] += lowpass_filter(np.random.normal(0, n_ratio, size=data[i,a:b,ind].shape), fc)
    
                
            if np.random.rand() < p_filt1:
                
                inds = np.random.choice(np.arange(12), size=np.random.randint(12), replace=False).ravel()
                for ind in inds:
                    fc = np.random.randint(13, 20)
                    data[i,:,ind] += lowpass_filter(data[i,:,ind], fc)
                
            elif np.random.rand() < p_filt2:
                
                inds = np.random.choice(np.arange(12), size=np.random.randint(12), replace=False).ravel()
                for ind in inds:
                    fc = np.random.randint(13, 20)
                    t_inds = np.sort(np.random.choice(np.arange(length), size=10, replace=False).ravel())
                    ai = np.random.randint(6)
                    a = t_inds[ai]
                    b = t_inds[np.random.randint(ai+3, 10)]
                    data[i,a:b,ind] += lowpass_filter(data[i,a:b,ind], fc)
        except:
            print('AUG ERROR!')
                            
        if np.random.rand() < p_shfl:
            inds = np.random.choice(np.arange(12), size=2, replace=False).ravel()
            tmp = data[i,:,inds[0]].copy()
            data[i,:,inds[0]] = data[i,:,inds[1]].copy()
            data[i,:,inds[1]] = tmp.copy()




        if np.random.rand() < p_scale:
            data[i] *= (0.5 + np.random.rand() * 1.)
        elif np.random.rand() < p_scale2:
            inds = np.random.choice(np.arange(12), size=np.random.randint(1,12), replace=False).ravel()
            for ind in inds:
                data[i,:,ind] *= (0.5 + np.random.rand() * 1.) 
                
        if np.random.rand() < p_shft:
            inds = np.random.choice(np.arange(12), size=np.random.randint(1,12), replace=False).ravel()
            for ind in inds:
                data[i,:,ind] += np.random.normal(0, 0.3)
        
        if np.random.rand() < p_inv1:
            ind = np.random.randint(12)
            data[i,:,ind] *= -1.

        elif np.random.rand() < p_inv2:
            inds = np.random.choice(np.arange(12), size=np.random.randint(2,12)).ravel()
            data[i,:,inds] *= -1.
            
        elif np.random.rand() < p_inv3:
            data[i] *= -1.
            
    return data


