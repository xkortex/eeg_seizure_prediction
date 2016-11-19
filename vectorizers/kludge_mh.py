from ..extern.seizure import transform

def vf_fft_timefreqcorr(data, verbose=True):
    
    process = transform.FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')
    # process = transform.FreqCorrelation(1, 48, 'usf', with_corr=True, with_eigen=True)
    outdata = process.apply(data.T) # transform wants time along axis 1
    if verbose: print(outdata.shape, type(outdata))
    return outdata