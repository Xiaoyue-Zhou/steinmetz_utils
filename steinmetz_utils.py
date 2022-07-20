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


    
def plot_by_pretrial_feedback(s, feedback_type, title = None, normalize = False) -> None:
    '''

    :param s: data of spike trians: (neurons * trials * time bins)
    :param feedback_type: an array made up of 1s & -1s. 1 for reward and -1 for penalty.
    :param title: the title of the plot generated. e.g., the name of brain region.
    :param normalize：whether normalize the spike data by trials and neurons.

    :return: plot (made by pyplot)

    '''

    isRew = feedback_type == 1
    isPen = feedback_type == -1

    lastIsRew = isRew[:-1]
    lastIsPen = isPen[:-1]

    if normalize:
        for i in range(s.shape[2]):
            s[:,:,i] = (s[:,:,i] - np.mean(s,2)) / np.std(s,2)

    s = s[:,1:,:]
    s_lastIsRew = s[:,lastIsRew,:]
    s_lastIsPen = s[:,lastIsPen,:]

    err_lastIsRew = np.std(np.mean(s_lastIsRew,0),0) / np.sqrt(s_lastIsRew.shape[1]-1)
    err_lastIsPen = np.std(np.mean(s_lastIsPen,0),0) / np.sqrt(s_lastIsPen.shape[1]-1)

    # draw the plot
    x = list(range(0,s.shape[2],1))
    plt.errorbar(x, np.mean(s_lastIsRew, (0,1)), yerr=err_lastIsRew, label='Post Reword')
    plt.errorbar(x, np.mean(s_lastIsPen, (0,1)), yerr=err_lastIsPen, label='Post Penalty')

    plt.title(title)
    plt.legend()
    plt.show()
