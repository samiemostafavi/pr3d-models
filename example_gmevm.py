import numpy as np
import json
from scipy.stats import norm
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import random

np.random.seed(0)
model_src_file = "ep5g-1/gmevm/model_0_parameters.json"

class GaussianExtremeValueMixtureModel:
    def __init__(self, weights, locations, scales, tail_threshold, tail_parameter, tail_scale):
        self.weights = weights
        self.locations = locations
        self.scales = scales
        self.tail_threshold = tail_threshold
        self.tail_parameter = tail_parameter
        self.tail_scale = tail_scale
        self.tail_cdf = self.cdf(self.tail_threshold)

    def logpdf(self, x):
        # Implement the logpdf function as needed for your application
        return np.log(self.pdf(x))

    def pdf(self, x):
        if x > self.tail_threshold:
            return genpareto.pdf(x, self.tail_parameter, self.tail_threshold, self.tail_scale)
        else:
            return self.weights @ np.array([norm.pdf(x, mu, sigma) for mu,sigma in zip(self.locations,self.scales)])
    
    def cdf(self, x):
        if x > self.tail_threshold:
            return genpareto.cdf(x, self.tail_parameter, self.tail_threshold, self.tail_scale)
        else:
            return self.weights @ np.array([norm.cdf(x, mu, sigma) for mu,sigma in zip(self.locations,self.scales)])

    def sample(self,num_samples):
        
        num_tail_samples = np.sum(np.random.rand(num_samples) > self.tail_cdf)
        num_bulk_samples = num_samples - num_tail_samples

        bulk_samples = np.concatenate([
            norm(loc = mu,scale = sigma).rvs(size=int(weight * num_bulk_samples)) for mu, sigma, weight in zip(self.locations, self.scales, self.weights)
        ])
        tail_samples = genpareto(self.tail_parameter, loc=self.tail_threshold, scale=self.tail_scale).rvs(size=num_tail_samples)
        return np.concatenate((bulk_samples, tail_samples))


# load json file
with open(model_src_file, 'r') as f:
    model_dict = json.load(f)

# create the Gaussian mixture model
gmevm = GaussianExtremeValueMixtureModel(
    model_dict["mixture_weights"],
    model_dict["mixture_locations"],
    model_dict["mixture_scales"],
    model_dict["tail_threshold"],
    model_dict["tail_parameter"],
    model_dict["tail_scale"]
)

# sample 1M
samples = gmevm.sample(100000)

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
plt.savefig('gmevm_result.png')  # Save the plot as res.png