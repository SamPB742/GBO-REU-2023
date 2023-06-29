# GBO-REU-2023
This codebase is designed to allow different kinds of analyses on RFI scans regularly performed by the 
Green Bank Telescope. These scans are accessible through a GUI that allows users to export details from 
scans in the form of a csv file with 'frequency', 'intensity', and 'scan__datetime' where all datapoints 
from a single scan have the same string value for the 'scan__datetime' field. 

Currently (Jun 29, 2023) the file rfi_scan_analysis/csv_utils.py can be run with the following options:
    - **integrate**: if the first command-line option is 'integrate', this should be followed by 
        - the filepath of the csv file to read the data from
        - the 'scan__datetime' of the scan to consider
        - the starting frequency in MHz
        - the ending frequency in MHz
        The code will integrate the intensity in that scan over the specified frequency range and print the
        result
    - **identify**: if the first command-line option is 'identify', this should be followed by 
        - the filepath of the csv file to read the data from
        - the 'scan__datetime' of the scan to consider
        - the starting frequency in MHz
        - the ending frequency in MHz
        The code will identify intensity peaks in the range, which are RFI features, and fit a gaussian 
        to each peak. It will then display the fit peaks over the actual data, alternating between green 
        and blue for different features (two peaks are considered to be in the same feature if the gaussian
        fit means are within 5 standard deviations of each other)
    - **compare**: if the first command-line option is 'compare', this should be followed by 
        - the filepath of the csv file to read the data from
        - the starting frequency in MHz
        - the ending frequency in MHz
        The code will consider all scans in the provided file and integrate the total intensity in each 
        scan from the start to end frequency, it will then graph the total intensity of that region over
        time by extracting the time of each scan from its 'scan__datetime' field