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

# Initialize the ETC, 'DEBUG' will allow you to see useful prints during the computation,
# skip_dataload = False will load the static sky configurations +  general transmissions
wst = WST(log = 'DEBUG', skip_dataload = False)

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

A full_obs dictionary should look like this:
```
full_obs = {
    "INS": "moslr",
    "CH": "red",
    
    "NDIT": 1,
    "DIT": 600, 
    
    "SNR": 5,
    "Lam_Ref": 5000,
    
    "OBJ_FIB_DISP": 0,
    
    "MOON": 'greysky',
    "PWV": 10,
    "FLI": 0.5,
    "SEE": 0.8,
    "AM": 1.2,
    "SKYCALC": False,
    
    "Obj_SED": 'template',
    "SED_Name": 'MARCS_8000K_lg+45',
    
    "OBJ_MAG": 15,
    "MAG_SYS": 'Vega',
    "MAG_FIL": 'V',
    
    "Z": 0,
    "BB_Temp": 9000.,
    "PL_Index": None,
    
    "SEL_FLUX": 50e-16,
    "SEL_CWAV": 8000,
    "SEL_FWHM":20,
    
    "Obj_Spat_Dis": 'resolved',
    
    "IMA": 'moffat',
    
    "IMA_FWHM": 0.5,
    "IMA_BETA": 2.5,
    
    "Sersic_Reff": 1,
    "Sersic_Ind": 3,
    
    "COADD_WL": 10,
    
    "COADD_XY": 1
}
```

After the computation results can be plotted easily accessing the mpdaf `Spectrum` objects in the results dictionaries like this:
```
res_snr['spec']['snr'].plot()
```
or
```
res_snr['spec']['nph_source'].plot()
```

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
