# pyetc_wst

Exposure Time Calculator (ETC) for the Wide-Field Spectroscopic Telescope (WST).

## Description

`pyetc_wst` is a Python package for exposure time calculation and signal-to-noise ratio (SNR) estimation for the WST instrument suite, including:

- **IFS** (Integral Field Spectrograph): Blue and Red channels
- **MOS-LR** (Multi-Object Spectrograph Low Resolution): Blue, Green, and Red channels  
- **MOS-HR** (Multi-Object Spectrograph High Resolution): U, B, V, and I channels

## Installation

### From GitHub

You can install directly from GitHub using pip:

```bash
pip install git+https://github.com/ferromatteo/pyetc_wst.git
```

### For Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ferromatteo/pyetc_wst.git
cd pyetc_wst
pip install -e .
```

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- astropy >= 5.0.0
- mpdaf >= 3.5.0
- skycalc_ipy >= 0.1.0

## Quick Start

```python
from pyetc_wst import WST

# Initialize the ETC
wst = WST()

# Display instrument information
wst.info()

# Access specific instruments
ifs_blue = wst.ifs['blue']
moslr_red = wst.moslr['red']
moshr_u = wst.moshr['U']

# Build the full dictionaries needed for computation (full_obs), which will include observing conditions, source properties, computation requests, and instrument configuration
full_obs = {...}
con, ob, spe, im, spe_input = wst.build_obs_full(full_obs)

# Compute time or snr given the full dictionary results

# for SNR:
res_snr = wst.snr_from_source(con, im, spe)

# for time/exposures/best combination
res_time = wst.time_from_source(con, im, spe, compute = 'dit'/'ndit'/'best')
```

Notebbok with specific examples will be included in future version

## Usage Examples

update in future version

## Documentation

update in future version

## Contributing

update in future version

## License

MIT License

## Citation

This package has been developed from the original `pyetc` package available at https://github.com/RolandBacon/pyetc

update in future version

## Contact

Matteo Ferro - [matteo.ferro@inaf.it]

Project Link: [https://github.com/ferromatteo/pyetc_wst](https://github.com/ferromatteo/pyetc_wst)
