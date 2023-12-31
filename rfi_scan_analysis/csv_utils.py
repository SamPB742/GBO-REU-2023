
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import stats
from scipy import optimize
from scipy import signal
from scipy import fft
import sys





"""
fpath1 = "/home/scratch/sbarton/GBO-REU-2023/rfi_scans_csv/test_hump.csv"
prime_import = pd.read_csv(fpath1)
fpath2 = "/home/scratch/sbarton/GBO-REU-2023/rfi_scans_csv/s_band.csv"
s_import = pd.read_csv(fpath2)
fpath3 = "/home/scratch/sbarton/GBO-REU-2023/rfi_scans_csv/c_band.csv"
c_import = pd.read_csv(fpath3)
fpath4 = "/home/scratch/sbarton/GBO-REU-2023/rfi_scans_csv/l_band.csv"
l_import = pd.read_csv(fpath4)
"""
#TODO comment out

"""
Class representing the fit to a Gaussian peak, stores the mean, std, and scale factor.
Init takes data and automatically fits and stores a gaussian in the new object.
GaussPeak.func allows querying the model gaussian at a particular input (frequency)
"""
class GaussPeak:
    """
    Fits a gaussian to the provided data and store it's parameters in this object

    Inputs: 
        data: the data to fit the gaussian to
        peak_freq: the frequency at which the peak we are fitting in the data occurs
        scan_name: the name of the scan in the dataset
    """
    def __init__(self, data, peak_idx, scan_name):
        right_min_idx, left_min_idx = find_limits(data, peak_idx)

        self.left_min_freq = data.loc[left_min_idx]['frequency']
        self.left_min_int = data.loc[left_min_idx]['intensity']
        self.right_min_freq = data.loc[right_min_idx]['frequency']
        self.right_min_int = data.loc[right_min_idx]['intensity']

        self.failed = False

        self.peak_data = trim_data(data, self.left_min_freq, self.right_min_freq, scan_name)

        params = cont_gauss_fit(self.peak_data)

        #self.yint = params[0]
        #self.slope = params[1]
        self.mean = params[0]
        self.std = params[1] 
        self.scale = params[2]

        #if fit mean is outside, mark this fit as failed
        if(self.mean < self.left_min_freq or self.mean > self.right_min_freq):
            self.failed = True
            self.mean = data.iloc[peak_idx]['frequency']
            #intentionally large std estimate avoids features being broken up by failed fits
            self.std = (self.right_min_freq - self.left_min_freq) / 3
            self.scale = 0 #failed peak will graph flat


        #this gaussian can be used to get values from this GaussPeak
        self.gauss = stats.norm(self.mean, self.std)

    def __str__(self):
        return f'Gaussian peak centered on {self.mean} with standard deviation of {self.std} and scaling factor {self.scale}'
    
    """
    Returns the value of this gaussian at the input frequency
    """
    def func(self, freq):
        return self.gauss.pdf(freq) * self.scale
    
    def plot(self, color):
        #generate freq chunks
        freqs = np.linspace(self.left_min_freq, self.right_min_freq, 100)
        #run through func
        plt.plot(freqs, self.func(freqs), color)
        #plot
    


class RFIFeature:
    """
    Creates a new RFI feature from a group of gaussian peaks

    Inputs:
        main_peak: a GaussPeak that is the central and/or largest peak in this feature
        sub_peaks: a list of other GaussPeaks in the same feature
    """
    def __init__(self, main_peak, all_peaks):
        if(not isinstance(main_peak, GaussPeak)):
                raise TypeError("Main peak must be a GaussPeak")
        self.main_peak = main_peak
        if(not isinstance(all_peaks, list) or not all(isinstance(item, GaussPeak) for item in all_peaks)):
            raise TypeError("Sub peaks must be a list of GaussPeak")
        self.all_peaks = all_peaks

    def plot(self, color):
        for peak in self.all_peaks:
            peak.plot(color)

    def __str__(self):
        string = "RFI Feature with main peak: " + str(self.main_peak) 
        string += "\n all peaks in this feature: \n"
        for peak in self.all_peaks:
            string += "\t" + str(peak) + "\n"
        return string
        
"""
Trim frequency-intensity data to only contain datapoints in a particular frequency range,
and optionally, from a particular scan. Additionally sorts the data points by frequency

Inputs: 
    data: a pandas dataframe with frequency-intensity information
    range_start: the beginning of the range in MHz
    range_end: the end of the range in MHz 
    scan_name: if provided, output will only contain data from this scan

Output:
    dataframe with only data between the start and end frequencies, and only
    from the particular scan (if specified), sorted by frequency
"""        
def trim_data(data, range_start, range_end, scan_name = None):
    data = data[(data['frequency'] >= range_start) & (data['frequency'] <= range_end)]
    if(scan_name != None):
        data = data[data['scan__datetime'] == scan_name]
    data = data.sort_values(by=['frequency'])
    if(data.size == 0):
        raise LookupError(f"No data in the requested region ({range_start} MHz - {range_end} MHz with scan name {scan_name}).")
    else:
        return data


"""
Integrates intensity over a frequency range  with scipy integrate

Inputs: 
    data: a pandas dataframe containing only data in the 
    frequency range to be integrated, sorted by frequency

Output:
    the integrated intensity over the region
"""
def integrate_range(data):
    result = integrate.trapezoid(data['intensity'], data['frequency'])
    #print(result)
    return result


""" 
Fits a gaussian profile to an intensity curve over a 
particular frequency range

Inputs: 
    data: a pandas dataframe containing only data from a particular scan, in the 
    frequency range to be fit, sorted by frequency

Output:
    the fit parameters:
        cont_yint: y-intercept of the continuum line 
        cont_slope: slope of the continuum line 
        gauss_mean: mean of the gaussian function 
        gauss_std: standard deviation of the gaussian function
        scan_name: name of the scan to pull from data
"""
def cont_gauss_fit(data):
    #starting guess for the mean in the middle of the frequency window
    mean_guess = data['frequency'].iloc[data['frequency'].size // 2]
    #since gaussian has area of 1, expect to scale by this much later
    scale_guess = integrate_range(data)
    #expect 6 sd lengths in the feature
    std_guess = (data['frequency'].size / 6) 

    #print(f'mean guess: {mean_guess}, std guess: {std_guess}, scale guess: {scale_guess}')
    #optimize 
    try:
        params = optimize.curve_fit(simple_gauss_func, data['frequency'], data['intensity'], (mean_guess, std_guess, scale_guess))[0]
    except RuntimeError as err:
        print(f'Failed to fit peak at {mean_guess}.')
        print(f'\t internal error {err}')
        return [mean_guess, std_guess, 0]
    except TypeError as err:
        print(f'Failed to fit peak at {mean_guess}.')
        print(f'\t internal error {err}')
        return [mean_guess, std_guess, 0]
    
    #print(params)
    
    """ old code for displaying the fit immediately, instead we would like to store it in the object
    #display the parameters
    print("mean: " + str(gauss_mean))
    print("std: " + str(gauss_std))
    print("yint: " + str(cont_yint))
    print("slope: " + str(cont_slope))
    print("scale: " + str(scale))
    #plot the data
    data_ax = plt.subplot(211)
    data_ax.plot('frequency', 'intensity', 'r', data=filtered_data, lw=1) #real data
    freq = filtered_data['frequency']
    intensity = filtered_data['intensity']
    data_ax.plot(freq, simple_gauss_func(freq, cont_yint, cont_slope, gauss_mean, gauss_std, scale), 'b', lw=1) #curve fit
    #residual subplot
    diff = intensity - simple_gauss_func(freq, cont_yint, cont_slope, gauss_mean, gauss_std, scale)
    residual_ax = plt.subplot(212, sharex=data_ax)
    residual_ax.plot(freq, diff, 'g.')
    """

    #plt.subplot(freq, diff, 'g')
    #plt.show()
    #store the fit params
    return params

"""
Function defining a gaussian profile plus linear continuum level,
has 5 free parameters to be fit to rfi scan features

Inputs:
    x: the frequency in MHz 
    
    gauss_mean: mean of the gaussian function 
    gauss_std: standard deviation of the gaussian function
    scale_factor: scale to multiply the whole distribution by

    These params have been removed
    cont_yint: y-intercept of the continuum line 
    cont_slope: slope of the continuum line 

Output: 
    The sum of a gaussian function with the given mean and std, scaled by the scale factor,
    plus a line with the given y-int and slope, evaluated at x
"""
def simple_gauss_func(x, gauss_mean, gauss_std, scale_factor):
    gaussian = stats.norm(gauss_mean, gauss_std)
    return (gaussian.pdf(x) * scale_factor)

"""
Function defining a triple gaussian profile plus linear continuum level,
has 11 free parameters to be fit to RFI scan features

The triple gaussian profile consists of one central gaussian and two 
other gaussians that are reflected across the center of the first
to mimic the symetrical 5-peaked profile often observed in RFI features

Inputs:
    x: the frequency in MHz 
    cont_yint: y-intercept of the continuum line 
    cont_slope: slope of the continuum line 
    central_mean: mean of the gaussian function 
    middle/outer_dist: distance of the middle/outer peaks from the central peaks
    central/middle/outer_std: standard deviation of each gaussian function
    central/middle/outer_scale: scale to multiply each gaussian function by

Output: 
    The sum of a gaussian function with the given mean and std, scaled by the scale factor,
    plus a line with the given y-int and slope, evaluated at x
"""
def triple_gauss_func(x, cont_yint, cont_slope, 
                      central_mean, central_std, central_scale, 
                      middle_dist, middle_std, middle_scale, 
                      outer_dist, outer_std, outer_scale):
    #make sure certain parameters are positive
    central_scale = np.abs(central_scale)
    middle_scale = np.abs(middle_scale)
    outer_scale = np.abs(outer_scale)
    middle_dist = np.abs(middle_dist)
    outer_dist = np.abs(outer_dist)


    central_gauss = stats.norm(central_mean, central_std)
    left_middle_gauss = stats.norm(central_mean - middle_dist, middle_std)
    left_outer_gauss = stats.norm(central_mean - outer_dist, outer_std)
    right_middle_gauss = stats.norm(central_mean + middle_dist, middle_std)
    right_outer_gauss = stats.norm(central_mean + outer_dist, outer_std)

    gauss_sum = (central_gauss.pdf(x) * central_scale)
    gauss_sum += (left_middle_gauss.pdf(x)  + right_middle_gauss.pdf(x)) * middle_scale
    gauss_sum += (left_outer_gauss.pdf(x) + right_outer_gauss.pdf(x)) * outer_scale

    return  gauss_sum







"""
Test function for integrated intensity composed of a gaussian 

def test_func(x, mean, std, scale_factor):
    #parameters for the test func for easy changing
    rv1 = stats.norm(mean, std)
    return (rv1.pdf(x) * scale_factor) 
    """
    
    #triple peaked gaussian for testing
"""
    central_mean = 0
    central_std = .2
    central_scale = 1
    middle_dist = 2
    middle_std = .5
    middle_scale = .5
    outer_dist = 5
    outer_std = .1
    outer_scale = .1

    central_gauss = stats.norm(central_mean, central_std)
    left_middle_gauss = stats.norm(central_mean - middle_dist, middle_std)
    left_outer_gauss = stats.norm(central_mean - outer_dist, outer_std)
    right_middle_gauss = stats.norm(central_mean + middle_dist, middle_std)
    right_outer_gauss = stats.norm(central_mean + outer_dist, outer_std)

    gauss_sum = (central_gauss.pdf(x) * central_scale)
    gauss_sum += (left_middle_gauss.pdf(x) * middle_scale) + (right_middle_gauss.pdf(x) * middle_scale)
    gauss_sum += (left_outer_gauss.pdf(x) * outer_scale) + (right_outer_gauss.pdf(x) * outer_scale)
    line_val =  cont_slope * x + cont_yint
    """

    #return  gauss_sum + line_val 





"""
Generates test data over a frequency range from the test func
also adds gaussian noise

Inputs:
    noise_level: additive noise is pulled from a standard normal distribution
        and multiplied by this factor for each datapoint
"""
def gen_gauss_test_data(range_start, range_end, mean, std, scale, num_points, noise_level):
    interval = (range_end - range_start) / num_points
    test_func = stats.norm(mean, std)
    #generate the x (frequency) values
    freq_data = []
    for num in range(num_points):
        freq_data.append(range_start + (interval * num))
        #freq_data.append(range_start + (interval * num))
    intensity_data = []
    #counter = 0
    for freq in freq_data:
        #if counter % 2 == 0 :
        intensity_data.append(test_func.pdf(freq) * scale)
        #else :
            #intensity_data.append(0)
        #counter += 1
    df = pd.DataFrame({'scan__datetime':'test_data', 'frequency': freq_data, 'intensity': intensity_data})
    noise = stats.norm().rvs(size=num_points) * noise_level


    df['intensity'] = df['intensity'] + noise

    return df 


#test_data = gen_test_data(-10, 10, 1000, .1)

#integrate_range(trim_data(test_data, -3, 3, '2022-07-01 00:08:38.400005+00:00'))



#test_fit = cont_gauss_fit(trim_data(test_data, -10, 10, '2022-07-01 00:08:38.400005+00:00'))

"""

cont_gauss_fit(prime_import, 850, 870, '2022-07-06 10:07:40.799982+00:00')

cont_gauss_fit(prime_import, 855.2, 855.5, '2022-07-06 10:07:40.799982+00:00')

cont_gauss_fit(prime_import, 857, 857.6, '2022-07-06 10:07:40.799982+00:00')

cont_gauss_fit(prime_import, 858, 858.6, '2022-07-06 10:07:40.799982+00:00')

cont_gauss_fit(s_import, 2700, 3000, '2023-03-12 22:37:55.199997+00:00')
"""


#cont_gauss_fit(c_import, 7410, 7420, '2023-03-25 17:55:40.799998+00:00', False)

#cont_gauss_fit(l_import, 1200, 1800, '2023-01-03 06:44:38.399996+00:00', False)

#cont_gauss_fit(l_import, 1306.45, 1306.6, '2023-01-03 06:44:38.399996+00:00', False)

#cont_gauss_fit(l_import, 1308.45, 1308.6, '2023-01-03 06:44:38.399996+00:00', False)

#cont_gauss_fit(l_import, 1261.48, 1261.58, '2023-01-03 06:44:38.399996+00:00', False)

#cont_gauss_fit(l_import, 1261.506, 1261.523, '2023-01-03 06:44:38.399996+00:00', True)

#cont_gauss_fit(l_import, 1255.9, 1256.1, '2023-01-03 06:44:38.399996+00:00', False)


"""
Takes the rolling average of a dataset

Inputs:
    a: a numerical data set
    window_size: the size of window to use for taking a rolling average, 
      should be smaller than len(a)

Outputs:
    a list of the rolling averages, of the same size as the input list
    there will be some edge effects visible
"""
def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'same'))

"""
Finds all the peaks in a data set and fit gaussian curves to them

Inputs:
    data: a pandas dataframe containing only data from a particular scan, in the 
    frequency range to be fit, sorted by frequency

Output:
    a list of RFIFeature objects representing the found features
"""
def fit_peaks(data):
    scan_name = data['scan__datetime'].iloc[0]
    #scan is already cont subtracted so mean is 0, values are already difference from mean
    data['smoothed'] = window_rms(data['intensity'], 5)
    noise_level = 5 * np.median(data['smoothed'])
    print(noise_level)
    peak_idxs = signal.find_peaks(data['intensity'], height=noise_level)[0] 
    #plt.plot('frequency', 'intensity', 'r', data=data, lw=1) #real data
    print(data.iloc[peak_idxs])
    crit_freqs = []
    crit_ints = []
    gauss_peaks = []
    #last_peak = None 
    for peak in peak_idxs:
        #don't need to keep track of these once GaussPeak implemented
        #crit_freqs.append(range_data['frequency'].iloc[peak])
        #crit_ints.append(range_data['intensity'].iloc[peak])

        #delegate fitting of feature to GaussPeak init
        this_peak = GaussPeak(data, peak, scan_name)

        gauss_peaks.append(this_peak)
        #last_peak = this_peak
        
        #crit_freqs.append(this_peak.left_min_freq)
        #crit_freqs.append(this_peak.right_min_freq)
        #crit_ints.append(this_peak.left_min_int)
        #crit_ints.append(this_peak.right_min_int)


    #plt.plot(crit_freqs, crit_ints, 'b.')
    gauss_features = []
    peak_set = [gauss_peaks[0]]

    last_mean = gauss_peaks[0].mean
    last_std = gauss_peaks[0].std
    main_peak = None
    for peak in gauss_peaks[1:]:
        #update main_peak if necessary
        #main peak is the tallest peak
        if(main_peak == None or main_peak.func(main_peak.mean) < peak.func(peak.mean)):
            main_peak = peak
        #start new feature if not close enough together
        if(last_mean + (5 * last_std) < peak.mean - (5 * peak.std)):
            gauss_features.append(RFIFeature(main_peak, peak_set))
            peak_set = [peak]
            main_peak = None
        else:
            peak_set.append(peak)
        """ code to plot individual peaks, no longer used
        #generate freq chunks
        freqs = np.linspace(peak.left_min_freq, peak.right_min_freq, 100)
        #run through func
        plt.plot(freqs, peak.func(freqs), color)
        #plot
        """
        last_mean = peak.mean
        last_std = peak.std

    if main_peak == None:
        main_peak = gauss_peaks[-1]
    #append last feature
    gauss_features.append(RFIFeature(main_peak, peak_set))

    #plot each feature
    

    for feature in gauss_features:
        print(feature)
    

    return gauss_features

"""
Finds the minima on either side of a peak

Inputs:
    data: a dataframe with frequency/intensity info from a scan
    peak_freq_idx: the index of the frequency peak we would like to find minima near

"""
def find_limits(data, peak_freq_idx):
    #go forward
    #print(data)
    right_min_idx = None
    left_min_idx = None
    last_int = data.iloc[peak_freq_idx]['intensity']
    last_idx = peak_freq_idx
    for idx, row in data.iloc[peak_freq_idx:].iterrows():
        if (row['intensity'] > last_int):
            right_min_idx = last_idx
            break
        last_int = row['intensity']
        last_idx = idx
        #print("row:" + str(freq) + "\n\n" + str(int))
    #print("right min idx for " + str(peak_freq_idx) + ": " + str(right_min_idx))
    last_int = data.iloc[peak_freq_idx]['intensity']
    last_idx = peak_freq_idx
    for idx, row in data.iloc[peak_freq_idx::-1].iterrows():
        if (row['intensity'] > last_int):
            left_min_idx = last_idx
            break
        last_int = row['intensity']
        last_idx = idx
    #print("left min idx for " + str(peak_freq_idx) + ": " + str(left_min_idx))

    #print(f"found limits {right_min_idx}, {left_min_idx}")

    return right_min_idx, left_min_idx




#TODO comment out
#fit_peaks(trim_data(l_import, 1400, 1500, '2023-01-03 06:44:38.399996+00:00'))
#fit_peaks(trim_data(s_import, 1800, 2600, '2023-03-12 22:37:55.199997+00:00'))

"""
Perform an FFT on an RFI scan and plot the results. 

Inputs:
    data: a pandas dataframe containing only data from a particular scan, in the 
    frequency range to be transformed, sorted by frequency
    

def fourier_transform(data):
    # Number of sample points
    N = data['intensity'].size
    # sample spacing
    T = N/1 #(range_end - range_start) #todo calculate these
    x = np.linspace(0.0, N*T, N, endpoint=False)
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = fft.fft(data['intensity'].to_numpy())
    xf = fft.fftfreq(N, T)[:N//2]
    import matplotlib.pyplot as plt
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()
"""



#fourier_transform(l_import, 1200, 1800, '2023-01-03 06:44:38.399996+00:00')
#fit_peaks(l_import, 1306.45, 1306.6, '2023-01-03 06:44:38.399996+00:00')


""" #TODO allow comparing mean and std as well
Compared integrated intensity of a frequency region over multiple scans

Inputs:
    data: a pandas dataframe containing data from an arbitrary number of scans, in the 
    frequency range to be integrated, sorted by frequency
    stat: either 'mean', 'std', 'num_peaks' or 'intensity' to determine what stat is 
    being compared
"""
def compare_scans(data, stat, start, end):
    uniques = data['scan__datetime'].unique()
    uniques = np.sort(uniques)
    scan_dates = []
    scan_datapoints = []
    for scan_name in uniques:
        scan_date = datetime.strptime(scan_name[0:19], '%Y-%m-%d %H:%M:%S')
        if stat == 'intensity':
            total = integrate_range(data[data['scan__datetime'] == scan_name])
            scan_datapoints.append(total)
            scan_dates.append(scan_date)
        elif stat == 'mean' or stat == 'std' or stat == 'num_peaks': 
            features = fit_peaks(trim_data(data, start, end, scan_name)) 
            if stat == 'mean':
                for feature in features:
                    if not feature.main_peak.failed:
                        scan_datapoints.append(feature.main_peak.mean)
                        scan_dates.append(scan_date)
            elif stat == 'std':
                for feature in features:
                    if not feature.main_peak.failed:
                        scan_datapoints.append(feature.main_peak.std)
                        scan_dates.append(scan_date)
            elif stat == 'num_peaks':
                total_peaks = 0
                for feature in features:
                    total_peaks += len(feature.all_peaks)
                scan_datapoints.append(total_peaks)
                scan_dates.append(scan_date)
        else:
            print(f"Statistic to be compared must be 'mean', 'std', 'num_peaks' or 'intensity', received {stat}")
            return

    plt.plot(scan_dates, scan_datapoints, 'r.')
    plt.show()
    return

#compare_scans(trim_data(l_import, 1425, 1430))

























#start copied code

def make_plot(data):
    # make a new object with the average intensity for the 2D plot
    mean_data_intens = data.groupby(
        ["scan__datetime", "frequency"]
    ).agg({"intensity": ["mean"]})
    mean_data_intens.columns = ["intensity_mean"]
    mean_data = mean_data_intens.reset_index()
    # sort values so the plot looks better, this has nothing to do with the actual data
    sorted_mean_data = mean_data.sort_values(by=["frequency", "intensity_mean"])

    # generate the description fro the plot
    txt = f" \
        Your data summary for this plot: \n \
        Receiver : test \n \
        Date range : From test to test \n \
        Frequency Range : {mean_data['frequency'].min()}MHz to {mean_data['frequency'].max()}MHz "

    # print out info for investagative GBO scientists
    print("Your requested projects are below:")
    print("Session Date \t\t Project_ID")
    print("-------------------------------------")
    #sort_by_date = sorted_mean_data.sort_values(by=["scan__session__name"])
    #project_ids = sort_by_date["scan__session__name"].unique()
    """for i in project_ids:
        proj_date = sort_by_date[
            sort_by_date["scan__session__name"] == i
        ].scan__datetime.unique()
        proj_date = proj_date.strftime("%Y-%m-%d")
        print(f"", proj_date[0], "\t\t", str(i))"""

    # Plot the 2D graph
    plt.figure(figsize=(9, 4))
    plt.title(txt, fontsize=8)
    plt.suptitle("Averaged RFI Environment at Green Bank Observatory")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Average Intensity (Jy)")
    plt.ylim(-10, 10)
    plt.plot(
        sorted_mean_data["frequency"],
        sorted_mean_data["intensity_mean"],
        color="black",
        linewidth=0.5,
    )
    # make sure the titles align correctly
    plt.tight_layout()
    # setting the location of the window
    mngr = plt.get_current_fig_manager()
    #geom = mngr.window.geometry()
    #x, y, dx, dy = geom.getRect()
    # display the plot to the right of the ui
    #mngr.window.setGeometry(459, 0, dx, dy)
    plt.show()

#end copied code

#print(data[data['intensity'] < 0])

#print(integrate_range(trim_data(prime_import, 840, 875, '2022-07-01 00:08:38.400005+00:00')))
#compare_scans(trim_data(l_import, 1425, 1430))
#fit_peaks(trim_data(l_import, 1200, 1800, '2023-01-03 06:44:38.399996+00:00'))

print("Hello World!")

#code for running from command line
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Expected command line args startng with integrate, identify, or compare")
    elif args[0] == "integrate":
        if len(args) != 5:
            print("Expected 5 arguments: integrate <filepath> <scan_name> <lower bound (MHz)> <upper bound (MHz)>")
        else:
            print(f"Integrating scan {args[2]} from file {args[1]} over the frequency range {args[3]} MHz to {args[4]} MHz")
            try:
                data = pd.read_csv(args[1])
            except RuntimeError as err:
                print("Failed to read csv file with error: " + err)
            try:
                result = integrate_range(trim_data(data, float(args[3]), float(args[4]), args[2]))
                print(result)
            except RuntimeError as err:
                print("Failed to integrate range with error: " + err)
            print(result)
    elif args[0] == "identify":
        if len(args) != 5:
            print("Expected 5 arguments: identify <filepath> <scan_name> <lower bound (MHz)> <upper bound (MHz)>")
        else:
            print(f"Identifying peaks in scan {args[2]} from file {args[1]} over the frequency range {args[3]} MHz to {args[4]} MHz")
            try:
                data = pd.read_csv(args[1])
            except RuntimeError as err:
                print("Failed to read csv file with error: " + err)
            try:
                trimmed_data = trim_data(data, float(args[3]), float(args[4]), args[2])
                result = fit_peaks(trimmed_data)
                #plot results
                plt.plot('frequency', 'intensity', 'r', data=trimmed_data, lw=1) #real data
                color = 'b'
                for feature in result:
                    #print(color)
                    #print(feature)
                    #print(feature.all_peaks)
                    
                    #switch color
                    if color == 'b':
                        color = 'g'
                    else:
                        color = 'b'

                    feature.plot(color)
                plt.show()
            except RuntimeError as err:
                print("Failed to fit peaks with error: " + err)
    elif args[0] == "compare":
        if len(args) != 5:
            print("Expected 5 arguments: compare <mean/std/intensity> <filepath> <lower bound (MHz)> <upper bound (MHz)>")
        else: 
            print(f"Comparing {args[1]} over the frequency range {args[3]} MHz to {args[4]} MHz for all scans in {args[2]}")
            try:
                data = pd.read_csv(args[2])
            except RuntimeError as err:
                print("Failed to read csv file with error: " + err)
            try: #pass bounds twice so can be trimmed again if necessary in compare_scans
                result = compare_scans(trim_data(data, float(args[3]), float(args[4])), args[1], float(args[3]), float(args[4]))
            except RuntimeError as err:
                print("Failed to compare scans with error: " + err)
    else:
        print(f"Invalid input {args[0]}, first argument must be integrate, identify, or compare")
