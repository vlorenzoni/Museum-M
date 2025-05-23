import numpy as np
import matplotlib.pyplot as plt

# read csv file
# read csv file

# read csv file
speakersFile = 'speakers_layout_GENELEC_v1.csv'
data = np.genfromtxt(speakersFile, delimiter=',', skip_header=1, dtype=float)


# extract columns

x = data[:, 0]  # x-coordinates
print(x)
y = data[:, 1]  # y-coordinates
print(y)
z = data[:, 2]  # z-coordinates
print(z)
#atan
angles = np.atan2(y, x)  # Calculate angles in radians
angles = np.degrees(angles)  # Convert to degrees
print(angles)





