import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

working_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'

permutation_filepath = working_dir + '/sex acceleration distribution.txt'

sex_acceleration_dist = np.loadtxt(permutation_filepath)

newlength = len(sex_acceleration_dist) - 1

indexes_to_keep = list(range(0, newlength))

new_sex_acc_dist = sex_acceleration_dist[indexes_to_keep]

empirical_sex_diff = sex_acceleration_dist[len(sex_acceleration_dist)-1]

# Calculte out what percentile value is with respect to arr
percentile = percentileofscore(sex_acceleration_dist, empirical_sex_diff)

plt.figure(figsize = (8, 8))
plt.hist(new_sex_acc_dist, bins = 10)
plt.title(f'Distribution of Sex Age Acceleration Difference(female - male)\nfrom {newlength} Permutation of Labels\n'
          f'Percentile of Empirical Value = {percentile:.1f}', fontsize=14)
plt.xlabel('Difference between Female and Male Acceleration in Age (years)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.axvline(x=empirical_sex_diff, color='r')
plt.show()

mystop=1