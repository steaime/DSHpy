import os
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
from DSH import SharedFunctions as sf   # <-- available on: https://github.com/steaime/DSHpy

if importlib.util.find_spec('cvxpy') is not None:
    import cvxpy as cp
else:
    print('cvxpy not installed. Pseudo-contin inversion will not work.')

def get_fpath(froot, angle_idx, rep_idx, fname_pre='', fname_ext='.ASC'):
    return os.path.join(froot, fname_pre+str(angle_idx).zfill(4)+'_'+str(rep_idx).zfill(4)+fname_ext)

def browse_folder(froot, fname_pre='', fname_ext='.ASC'):
    corr_fnames = sf.FindFileNames(froot, Prefix=fname_pre, Ext=fname_ext, ExcludeStrings=['_averaged'], Sort='ASC')
    angle_id = [sf.AllIntInStr(f)[-2] for f in corr_fnames]
    unique_angleIDs = list(set(angle_id))
    unique_angleIDs.sort()
    nrep = np.max([sf.LastIntInStr(f) for f in corr_fnames])
    return corr_fnames, unique_angleIDs, nrep

def read_file(fpath, skiplines=15, corr_hdr='"Correlation"', count_hdr='"Count Rate"', key_split=':', max_count=1000, plot=False):
    res = {}
    #print(fpath)
    with open(fpath, encoding="latin-1") as f:
        for i in range(skiplines):
            temp_str = f.readline().strip()
        while (temp_str[:len(corr_hdr)]!=corr_hdr):
            str_split = temp_str.split(key_split)
            if (len(str_split)>1):
                temp_val = str_split[1].strip()
                try:
                    temp_val = float(temp_val)
                except:
                    pass
                res[str_split[0].strip()] = temp_val
            temp_str = f.readline().strip()
        res['theta'] = res['Angle [°]'] * np.pi/180.
        res['q'] = (4*np.pi*res['Refractive Index']/(1e-3*res['Wavelength [nm]']))*np.sin(res['theta']/2)
        corr_data = []
        while (temp_str[:len(count_hdr)]!=count_hdr):
            temp_str = f.readline().strip()
            str_split = temp_str.split('\t')
            if len(str_split)>1:
                tmp_row = []
                try:
                    for s in str_split:
                        tmp_row.append(float(s.strip()))
                except:
                    print(str_split)
                    tmp_row = None
            if tmp_row is not None:
                corr_data.append(tmp_row)
        res['corr'] = np.asarray(corr_data)
        count_data = []
        while len(count_data)<max_count:
            temp_str = f.readline().strip()
            str_split = temp_str.split('\t')
            if len(str_split)>1:
                tmp_row = []
                try:
                    for s in str_split:
                        tmp_row.append(float(s.strip()))
                except:
                    break
            count_data.append(tmp_row)
        res['count'] = np.asarray(count_data)
        if plot:
            fig, ax = plt.subplots(figsize=(10,10), nrows=2)
            ax[0].plot(res['corr'][:,0], res['corr'][:,1], '.', label='CH0')
            ax[0].plot(res['corr'][:,0], res['corr'][:,2], '.', label='CH1')
            ax[0].set_xscale('log')
            ax[0].set_xlabel(r'$\tau$ [ms]')
            ax[0].set_ylabel(r'$g_2(q, \tau) - 1$')
            ax[0].legend()
            ax[1].plot(res['count'][:,0], res['count'][:,1], label='CH0')
            ax[1].plot(res['count'][:,0], res['count'][:,2], label='CH1')
            ax[1].set_xlabel(r'$t$ [s]')
            ax[1].set_ylabel(r'$I(q, t)$')
            ax[1].legend()
            res['plot'] = fig
    return res

def merge_results(froot, fname_pre='', fname_ext='.ASC', plot=True):

    corr_fnames, unique_angleIDs, nrep = browse_folder(froot, fname_pre, fname_ext)

    dt, q, avg_corr, std_corr = [], [], [], []
    exptau, expfit_corr = [], []
    expfit_ylim = 0.7
    num_tau = -1

    for aid in unique_angleIDs:
        ccorr = []
        for i in range(nrep):
            if os.path.isfile(get_fpath(froot,aid,i+1, fname_pre, fname_ext)):
                cur_data = read_file(get_fpath(froot,aid,i+1, fname_pre, fname_ext))
                add = True
                if num_tau > 0:
                    if cur_data['corr'].shape[0] != num_tau:
                        print('File {0} is corrupted: correlation function has {1} points instead of {2}'.format(get_fpath(aid,i+1), cur_data['corr'].shape[0], num_tau))
                        add = False
                else:
                    num_tau = cur_data['corr'].shape[0]
                    dt = cur_data['corr'][:,0]
                if add:
                    ccorr.append(cur_data['corr'][:,1])
                    ccorr.append(cur_data['corr'][:,2])
        q.append(cur_data['q'])
        cur_avg = np.mean(ccorr, axis=0)
        avg_corr.append(cur_avg)
        std_corr.append(np.std(ccorr, axis=0))

        expfit_select = np.where(cur_avg/cur_avg[0] > expfit_ylim)
        x_expfit = dt[expfit_select]
        y_expfit = np.log(cur_avg[expfit_select]/cur_avg[0])
        expfitp = np.polyfit(x_expfit, y_expfit, 1)
        expfit_corr.append(np.exp(np.poly1d(expfitp)(dt)))
        exptau.append(-2.0/expfitp[0])
        
    if plot:
        colors = plt.cm.jet(np.linspace(0,1,len(avg_corr)))
            
        fig, ax = plt.subplots(figsize=(12,21), nrows=3)
        for i in range(len(avg_corr)):
            ax[0].errorbar(dt, avg_corr[i], yerr=std_corr[i], fmt='.', color=colors[i], label='{0:.1f}'.format(q[i]))
            ax[1].errorbar(dt, avg_corr[i]/avg_corr[i][0], yerr=std_corr[i]/avg_corr[i][0], fmt='.', color=colors[i], label='{0:.1f}'.format(q[i]))
            ax[1].plot(dt, expfit_corr[i], ':', color=colors[i])
            ax[2].errorbar(dt, avg_corr[i]/avg_corr[i][0], yerr=std_corr[i]/avg_corr[i][0], fmt='.', color=colors[i], label='{0:.1f}'.format(q[i]))
            ax[2].plot(dt, expfit_corr[i], ':', color=colors[i])

        for i in [0,1]:
            ax[i].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].set_xlim([0,10])
        ax[2].set_ylim([0.01,1])
        ax[-1].set_xlabel(r'$\tau$ [ms]')
        ax[0].set_ylabel(r'$g_2(q, \tau) - 1$')
        for i in [1,2]:
            ax[i].set_ylabel(r'$\left[ g_2(q, \tau) - 1 \right]/\beta$')
        ax[1].legend()

        fig.savefig(os.path.join(froot, 'CorrFunctions_exp.png'))

    return {'tau': np.array(dt), 'q': np.array(q), 'avg_corr': np.array(avg_corr), 'std_corr': np.array(std_corr), 
            'exptau': np.array(exptau), 'expfit_corr': np.array(expfit_corr)}

def pseudo_contin(res, plot=True):

    if importlib.util.find_spec('cvxpy') is not None:
        Gamma = np.logspace(-4, 3, 300) # bins for decay rates (units of 1/ms)
        K = np.exp(-np.outer(res['tau'], Gamma)) # kernel matrix for discretized Laplace transform
        L = np.diff(np.eye(len(Gamma)), 2, axis=0) # smoothness operator
        alpha = 1e-2 # regularization parameter
        x = cp.Variable(len(Gamma), nonneg=True) # optimization variable

        Gamma_distr = []
        for i in range(len(res['avg_corr'])):
            objective = cp.Minimize(
                cp.sum_squares(K @ x - res['avg_corr'][i])
                + alpha * cp.sum_squares(L @ x)
            )

            problem = cp.Problem(objective)
            problem.solve()
            Gamma_distr.append(x.value)

        if plot:
            fig, ax = plt.subplots(figsize=(10,6), ncols=2)
            fig.suptitle("CONTIN-like inversion")
            for i in range(len(Gamma_distr)):
                ax[0].loglog(Gamma, Gamma_distr[i], label='{0:.1f}'.format(res['q'][i]))
                ax[1].loglog(Gamma/res['q'][i]**2, Gamma_distr[i], label='{0:.1f}'.format(res['q'][i]))
            ax[0].set_xlabel(r"Decay rate $\Gamma[ms^{-1}]$ ")
            ax[0].set_ylabel("Amplitude")
            ax[1].set_xlabel(r"$\Gamma/q^2 [\mu m^2/ms]$")
            ax[0].legend()
            for cax in ax:
                cax.set_ylim([1e-3, 1])
            ax[1].set_xlim([1e-6, 1])
            fig.tight_layout()

        return Gamma_distr
    
    else:

        raise ImportError("cvxpy is not installed. Pseudo-contin inversion cannot be performed.")