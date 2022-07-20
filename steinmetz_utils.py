import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def align_to_gocue(spks: np.ndarray, gocues: np.ndarray, bins=100) -> np.ndarray: # spks: ( 734, 214, 250 )
    gocues = gocues.squeeze() # ( 214, )

    spks_aligned = []
    for trial in range(len(gocues)):
        gocue = int(gocues[trial] * 100)

        spk = spks[:, trial, gocue:gocue+bins] # ( 734, 100 )
        assert spk.shape[-1] == bins
        
        spks_aligned.append(spk)

    
    spks_aligned = np.stack(spks_aligned).transpose(1,0,2)

    return spks_aligned # ( 734, 214, 100 )


def split_to_regions(data: np.ndarray, regions: np.ndarray) -> dict:
    regions_unique = np.unique(regions)

    regions_idxs = [np.where(regions == region)[0] for region in regions_unique] # unwise to use np.where but I forgot 

    data_dict = {}
    for idxs, name in zip(regions_idxs, regions_unique):
        data_dict.update({name: data[idxs]})

    return data_dict



def timecourse_3d(timecourse: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(*timecourse)
    
 
def neuronModel_thisTrialResp(alldat, session, brain_area, till=50,
                              penalty='l2', C=1, cv=16):
    '''

    :param alldat: Steinmetz dataset
    :param session: select from 0~38
    :param brain_area: needed brain area name. e.g.,'VISa'
    :param till: the number of time bins needed.
    :param penalty: 'l1'/'l2' type of penalty for regularization. set to 'l2' by default
    :param C: strength of penalty if any penalty. C=1/lambda.
    :param cv: number of folds when doing the cross validation test.

    :return: ori_acc(train set acc) & ori_cvd(test set acc)

    '''

    dat = alldat[session]

    isResp = dat['response'] != 0
    isNeededArea = dat['brain_area'] == brain_area

    respAll = dat['response']
    resp = respAll[isResp]     # '-1' means the right stimulus has higher contrast
    resp = resp.astype('int')

    s = dat['spks']
    s = s[:,isResp,:till]
    s = s[isNeededArea,:,:]

    ## compute firing rate - average
    fRate = np.sum(s,2)
    fRate = fRate.T
    dim = fRate.shape[0]
    # add 1s line
    fRate = np.concatenate((np.ones(fRate.shape[0]).reshape(dim,1),fRate), axis=1)

    # fit GLM with regularization
    if penalty == 'l2':
        model = LogisticRegression(penalty='l2',C=C,max_iter=5000)
    elif penalty == 'l1':
        model = LogisticRegression(penalty='l1',C=C,max_iter=5000,solver='saga')
    else:
        model = LogisticRegression(penalty=None, max_iter=5000)

    model.fit(fRate,resp)

    ori_acc = compute_acc(fRate, resp, model) # a single number
    ori_cvd = cross_val_score(model, fRate, resp, cv=cv) # an array (cv,)

    return ori_acc,  ori_cvd
