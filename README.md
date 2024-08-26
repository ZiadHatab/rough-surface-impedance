# Surface Impedance

Implementation of the Gradient model of [1] to describe surface impedance of conductors with roughness.

You can use different probability distributions for the roughness. I included Normal, Rayleigh, and Uniform. Read the comments in the functions!

## Installation

You need [Scipy](https://scipy.org/) and [Numpy](https://numpy.org/) for computation, and [Matplotlib](https://matplotlib.org/) for plotting.

## Example

Recreating the example plots from [1].

![B-field plot](images/B-field_plot.png)

![Surface Impedance](images/surface_impedance.png)

## References

[1] G. Gold and K. Helmreich, "A Physical Surface Roughness Model and Its Applications," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 10, pp. 3720-3732, 
Oct. 2017, doi: [10.1109/TMTT.2017.2695192](https://doi.org/10.1109/TMTT.2017.2695192).

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)