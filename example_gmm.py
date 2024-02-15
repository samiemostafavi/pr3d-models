import numpy as np
import json
from scipy.stats import norm
import matplotlib.pyplot as plt

model_src_file = "ep5g-1/gmm/model_0_parameters.json"

class GaussianMixtureModel:
    def __init__(self, weights, locations, scales):
        self.weights = weights
        self.locations = locations
        self.scales = scales

    def logpdf(self, x):
        # Implement the logpdf function as needed for your application
        return np.log(self.pdf(x))

    def pdf(self, x):
        # Implement the pdf function as needed for your application
        return self.weights @ np.array([norm.pdf(x, mu, sigma) for mu,sigma in zip(self.locations,self.scales)])
    
    def cdf(self, x):
        return self.weights @ np.array([norm.cdf(x, mu, sigma) for mu,sigma in zip(self.locations,self.scales)])

    def sample(self,num_samples):
        return np.concatenate([
            norm(loc = mu,scale = sigma).rvs(size=int(weight * num_samples)) for mu, sigma, weight in zip(self.locations, self.scales, self.weights)
        ])

# load json file
with open(model_src_file, 'r') as f:
    model_dict = json.load(f)

# create the Gaussian mixture model
gmm = GaussianMixtureModel(model_dict["mixture_weights"],model_dict["mixture_locations"],model_dict["mixture_scales"])

# sample 1M
samples = gmm.sample(100000)

# Plot CCDF of samples
samples_sorted = np.sort(samples)
ccdf = 1 - np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)
plt.plot(samples_sorted, ccdf)
plt.xscale('linear')  # Set x-axis scale to linear
plt.yscale('log')     # Set y-axis scale to log
plt.xlim(0, 30)       # Set x-axis limit to [0, 30]
plt.xlabel('Delay [ms]')
plt.ylabel('CCDF')
plt.title('Complementary Cumulative Distribution Function (CCDF)')
plt.grid(True)
plt.savefig('gmm_result.png')  # Save the plot as res.png