#!/usr/bin/env python3
"""Estimate primary-photon ambient dose equivalent H*(10), mrem/pulse.

Edit CONFIG or pass --config FILE.json. Python 3.10+, numpy, scipy required.
Al, SS, and Teflon (PTFE) targets; shield paths may also contain BK7 or BaF2. SS is a
simplified 304 alloy and BK7 is a documented nominal-composition surrogate.
The target is assumed electron-stopping (thick). SS and Teflon source yields
are approximate mass-weighted elemental-yield surrogates. No target self-absorption,
scattered radiation, secondary particles, or neutron dose is calculated.
See dose_calculator.tex for equations, tables and limitations.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, quad

CONFIG = {
    # "laser": {
    #     "pulse_energy": 20,
    #     "pulse_duration": 2e-12,  # intensity FWHM
    #     "wavelength": 9.3e-6,
    #     "beam_diameter": 25.0e-3,  # incident intensity FWHM at lens
    #     "focal_length": 1.0,
    #     "M_squared": 1.0,
    # },
    
    # "laser": {
    #     "pulse_energy": 20,
    #     "pulse_duration": 70e-12,  # intensity FWHM
    #     "wavelength": 9.3e-6,
    #     "beam_diameter": 38.0e-3,  # incident intensity FWHM at lens
    #     "focal_length": 1.0,
    #     "M_squared": 1.0,
    # },
    
    "laser": {
        "pulse_energy": 14,
        "pulse_duration": 2e-12,  # intensity FWHM
        "wavelength": 9.2e-6,
        "beam_diameter": 38.0e-3,  # incident intensity FWHM at lens
        "focal_length": 4.5,
        "M_squared": 1.0,
    },
    
    
    "target": "Teflon",  # "Al", "SS", or "Teflon" (PTFE)
    "source": {
        "observation_angle": 0.0,
    },
    # Ordered ray segments from source to the single observation point.
    # Every segment, including vacuum, contributes to inverse-square distance.
    # These are lengths ALONG THE OBSERVATION RAY, not necessarily normal thicknesses.
    "path": [
        {"material": "vacuum", "thickness": 0.300},
        {"material": "SS", "thickness": 0.0127},
        #{"material": "air", "thickness": 0.0},
    ],
}

# Advanced model constants. Normal users should not need to change these.
SOURCE_MODEL = {
    "laser_to_hot_electron_fraction": 1.0,
    "temperature_keV": None,  # None = laser scaling; positive value overrides
    "angular_sigma_rad": 0.7853981633974483,
}
DENSITY = {
    "Al": 2699.0, "SS": 7900.0, "air": 1.2048,
    "BK7": 2510.0, "BaF2": 4890.0,
}

KEV_TO_J = 1.602176634e-16
REST_KEV = 511.0
K_MIN, K_MAX = 10.0, 20000.0
SS_WEIGHTS = {"Fe": 0.738, "Cr": 0.181, "Ni": 0.081}
TEFLON_WEIGHTS = {"C": 0.240183, "F": 0.759817}  # NIST PTFE mass fractions
BK7_WEIGHTS = {
    "Si": 0.3231389993841853, "O": 0.48388281234709274,
    "B": 0.03338480543545398, "Ba": 0.02749656874894017,
    "Na": 0.07715367482585571, "K": 0.052216516907371474,
    "As": 0.002726622351100653,
}
BAF2_WEIGHTS = {"Ba": 0.7832764008365862, "F": 0.2167235991634137}
ATOMIC_Z = {"C": 6, "F": 9, "Al": 13, "Cr": 24, "Fe": 26, "Ni": 28}

# Embedded coefficient tables are inserted below. No network access at runtime.
NIST_TABLES = {'Al': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html',
        'rows': [[1, 1185],
                 [1.5, 402.2],
                 [1.5596, 362.1],
                 [1.5596, 3957],
                 [2, 2263],
                 [3, 788],
                 [4, 360.5],
                 [5, 193.4],
                 [6, 115.3],
                 [8, 50.33],
                 [10, 26.23],
                 [15, 7.955],
                 [20, 3.441],
                 [30, 1.128],
                 [40, 0.5685],
                 [50, 0.3681],
                 [60, 0.2778],
                 [80, 0.2018],
                 [100, 0.1704],
                 [150, 0.1378],
                 [200, 0.1223],
                 [300, 0.1042],
                 [400, 0.09276],
                 [500, 0.08445],
                 [600, 0.07802],
                 [800, 0.06841],
                 [1000, 0.06146],
                 [1250, 0.05496],
                 [1500, 0.05006],
                 [2000, 0.04324],
                 [3000, 0.03541],
                 [4000, 0.03106],
                 [5000, 0.02836],
                 [6000, 0.02655],
                 [8000, 0.02437],
                 [10000, 0.02318],
                 [15000, 0.02195],
                 [20000, 0.02168]],
        'Z': 13},
 'Cr': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z24.html',
        'rows': [[1, 7405],
                 [1.5, 2694],
                 [2, 1277],
                 [3, 433.9],
                 [4, 198.8],
                 [5, 108],
                 [5.9892, 65.74],
                 [5.9892, 597.7],
                 [6, 516],
                 [8, 251.3],
                 [10, 138.6],
                 [15, 45.71],
                 [20, 20.38],
                 [30, 6.434],
                 [40, 2.856],
                 [50, 1.55],
                 [60, 0.9639],
                 [80, 0.4905],
                 [100, 0.3166],
                 [150, 0.1788],
                 [200, 0.1378],
                 [300, 0.1067],
                 [400, 0.09213],
                 [500, 0.08281],
                 [600, 0.07598],
                 [800, 0.0662],
                 [1000, 0.0593],
                 [1250, 0.05295],
                 [1500, 0.04832],
                 [2000, 0.04213],
                 [3000, 0.03559],
                 [4000, 0.03235],
                 [5000, 0.03057],
                 [6000, 0.02956],
                 [8000, 0.02869],
                 [10000, 0.02855],
                 [15000, 0.0292],
                 [20000, 0.03026]],
        'Z': 24},
 'Fe': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html',
        'rows': [[1, 9085],
                 [1.5, 3399],
                 [2, 1626],
                 [3, 557.6],
                 [4, 256.7],
                 [5, 139.8],
                 [6, 84.84],
                 [7.112, 53.19],
                 [7.112, 407.6],
                 [8, 305.6],
                 [10, 170.6],
                 [15, 57.08],
                 [20, 25.68],
                 [30, 8.176],
                 [40, 3.629],
                 [50, 1.958],
                 [60, 1.205],
                 [80, 0.5952],
                 [100, 0.3717],
                 [150, 0.1964],
                 [200, 0.146],
                 [300, 0.1099],
                 [400, 0.094],
                 [500, 0.08414],
                 [600, 0.07704],
                 [800, 0.06699],
                 [1000, 0.05995],
                 [1250, 0.0535],
                 [1500, 0.04883],
                 [2000, 0.04265],
                 [3000, 0.03621],
                 [4000, 0.03312],
                 [5000, 0.03146],
                 [6000, 0.03057],
                 [8000, 0.02991],
                 [10000, 0.02994],
                 [15000, 0.03092],
                 [20000, 0.03224]],
        'Z': 26},
 'Ni': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z28.html',
        'rows': [[1, 9855],
                 [1.00404, 9753],
                 [1.0081, 9654],
                 [1.0081, 10990],
                 [1.5, 4234],
                 [2, 2049],
                 [3, 709.4],
                 [4, 328.2],
                 [5, 179.3],
                 [6, 109],
                 [8, 49.52],
                 [8.3328, 44.28],
                 [8.3328, 329.4],
                 [10, 209],
                 [15, 70.81],
                 [20, 32.2],
                 [30, 10.34],
                 [40, 4.6],
                 [50, 2.474],
                 [60, 1.512],
                 [80, 0.7306],
                 [100, 0.444],
                 [150, 0.2208],
                 [200, 0.1582],
                 [300, 0.1154],
                 [400, 0.09765],
                 [500, 0.08698],
                 [600, 0.07944],
                 [800, 0.06891],
                 [1000, 0.0616],
                 [1250, 0.05494],
                 [1500, 0.05015],
                 [2000, 0.04387],
                 [3000, 0.03745],
                 [4000, 0.03444],
                 [5000, 0.03289],
                 [6000, 0.0321],
                 [8000, 0.03164],
                 [10000, 0.03185],
                 [15000, 0.0332],
                 [20000, 0.03476]],
        'Z': 28},
 'air': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html',
         'rows': [[1, 3606],
                  [1.5, 1191],
                  [2, 527.9],
                  [3, 162.5],
                  [3.2029, 134],
                  [3.2029, 148.5],
                  [4, 77.88],
                  [5, 40.27],
                  [6, 23.41],
                  [8, 9.921],
                  [10, 5.12],
                  [15, 1.614],
                  [20, 0.7779],
                  [30, 0.3538],
                  [40, 0.2485],
                  [50, 0.208],
                  [60, 0.1875],
                  [80, 0.1662],
                  [100, 0.1541],
                  [150, 0.1356],
                  [200, 0.1233],
                  [300, 0.1067],
                  [400, 0.09549],
                  [500, 0.08712],
                  [600, 0.08055],
                  [800, 0.07074],
                  [1000, 0.06358],
                  [1250, 0.05687],
                  [1500, 0.05175],
                  [2000, 0.04447],
                  [3000, 0.03581],
                  [4000, 0.03079],
                  [5000, 0.02751],
                  [6000, 0.02522],
                  [8000, 0.02225],
                  [10000, 0.02045],
                  [15000, 0.0181],
                  [20000, 0.01705]]},
 'B': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z05.html',
       'rows': [[10,1.255],[15,.4827],[20,.3014],[30,.2063],[40,.1793],
                [50,.1665],[60,.1583],[80,.1472],[100,.1391],[150,.1243],
                [200,.1136],[300,.09862],[400,.08834],[500,.08065],
                [600,.07460],[800,.06549],[1000,.05890],[1250,.05266],
                [1500,.04791],[2000,.04108],[3000,.03284],[4000,.02798],
                [5000,.02476],[6000,.02248],[8000,.01945],[10000,.01755],
                [15000,.01495],[20000,.01368]]},
 'O': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z08.html',
       'rows': [[10,5.952],[15,1.836],[20,.8651],[30,.3779],[40,.2585],
                [50,.2132],[60,.1907],[80,.1678],[100,.1551],[150,.1361],
                [200,.1237],[300,.1070],[400,.09566],[500,.08729],
                [600,.08070],[800,.07087],[1000,.06372],[1250,.05697],
                [1500,.05185],[2000,.04459],[3000,.03597],[4000,.03100],
                [5000,.02777],[6000,.02552],[8000,.02263],[10000,.02089],
                [15000,.01866],[20000,.01770]]},
 'F': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z09.html',
       'rows': [[10,8.205],[15,2.492],[20,1.133],[30,.4487],[40,.2828],
                [50,.2214],[60,.1920],[80,.1639],[100,.1496],[150,.1298],
                [200,.1176],[300,.1015],[400,.09073],[500,.08274],
                [600,.07649],[800,.06717],[1000,.06037],[1250,.05399],
                [1500,.04915],[2000,.04228],[3000,.03422],[4000,.02960],
                [5000,.02663],[6000,.02457],[8000,.02195],[10000,.02039],
                [15000,.01846],[20000,.01769]]},
 'Na': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z11.html',
        'rows': [[10,15.57],[15,4.694],[20,2.057],[30,.7197],[40,.3969],
                 [50,.2804],[60,.2268],[80,.1796],[100,.1585],[150,.1335],
                 [200,.1199],[300,.1029],[400,.09185],[500,.08372],
                 [600,.07736],[800,.06788],[1000,.06100],[1250,.05454],
                 [1500,.04968],[2000,.04282],[3000,.03487],[4000,.03037],
                 [5000,.02753],[6000,.02559],[8000,.02319],[10000,.02181],
                 [15000,.02023],[20000,.01970]]},
 'Si': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z14.html',
        'rows': [[10,33.89],[15,10.34],[20,4.464],[30,1.436],[40,.7012],
                 [50,.4385],[60,.3207],[80,.2228],[100,.1835],[150,.1448],
                 [200,.1275],[300,.1082],[400,.09614],[500,.08748],
                 [600,.08077],[800,.07082],[1000,.06361],[1250,.05688],
                 [1500,.05183],[2000,.04480],[3000,.03678],[4000,.03240],
                 [5000,.02967],[6000,.02788],[8000,.02574],[10000,.02462],
                 [15000,.02352],[20000,.02338]]},
 'K': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z19.html',
       'rows': [[10,79.07],[15,25.03],[20,10.93],[30,3.413],[40,1.541],
                [50,.8679],[60,.5678],[80,.3251],[100,.2345],[150,.1582],
                [200,.1319],[300,.1080],[400,.09495],[500,.08600],
                [600,.07922],[800,.06929],[1000,.06216],[1250,.05556],
                [1500,.05068],[2000,.04399],[3000,.03666],[4000,.03282],
                [5000,.03054],[6000,.02915],[8000,.02766],[10000,.02704],
                [15000,.02687],[20000,.02737]]},
 'As': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z33.html',
        'rows': [[10,41.15],[11.8667,25.77],[11.8667,179.2],[15,98.56],
                 [20,45.64],[30,15.06],[40,6.760],[50,3.635],[60,2.203],
                 [80,1.030],[100,.5971],[150,.2622],[200,.1719],[300,.1150],
                 [400,.09414],[500,.08259],[600,.07483],[800,.06440],
                 [1000,.05735],[1250,.05106],[1500,.04661],[2000,.04093],
                 [3000,.03539],[4000,.03296],[5000,.03187],[6000,.03141],
                 [8000,.03146],[10000,.03207],[15000,.03405],[20000,.03603]]},
 'Ba': {'url': 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z56.html',
        'rows': [[10,186.0],[15,63.47],[20,29.38],[30,9.904],
                 [37.4406,5.498],[37.4406,29.19],[40,24.57],[50,13.79],
                 [60,8.511],[80,3.963],[100,2.196],[150,.7828],[200,.4046],
                 [300,.1891],[400,.1265],[500,.09923],[600,.08410],
                 [800,.06744],[1000,.05803],[1250,.05058],[1500,.04592],
                 [2000,.04078],[3000,.03692],[4000,.03598],[5000,.03612],
                 [6000,.03669],[8000,.03844],[10000,.04042],[15000,.04518],
                 [20000,.04902]]}}

HSTAR_FLUENCE_TABLE = [(10, 0.083),
 (15, 0.85),
 (20, 1.05),
 (30, 0.8),
 (40, 0.62),
 (50, 0.52),
 (60, 0.51),
 (80, 0.56),
 (100, 0.62),
 (150, 0.87),
 (200, 1.23),
 (300, 1.81),
 (400, 2.36),
 (500, 2.78),
 (600, 3.46),
 (800, 4.29),
 (1000, 5.18),
 (1500, 6.92),
 (2000, 8.25),
 (3000, 10.4),
 (4000, 10.7),
 (5000, 10.4),
 (6000, 9.58),
 (8000, 9.1),
 (10000, 8.76),
 (20000, 8.29)]


class LogTable:
    """Bounded log-log interpolation; repeated edge energies bracket a jump."""
    def __init__(self, rows, value_scale=1.0):
        a = np.asarray(rows, dtype=float)
        self.energy = a[:, 0].copy()
        self.values = a[:, 1].copy()*value_scale
        for i in np.flatnonzero(np.diff(self.energy) == 0):
            self.energy[i] *= 1 - 1e-8
            self.energy[i+1] *= 1 + 1e-8
        if np.any(np.diff(self.energy) <= 0) or np.any(a <= 0):
            raise ValueError("Invalid embedded coefficient table")

    def __call__(self, energy):
        e = np.asarray(energy, dtype=float)
        if (not np.all(np.isfinite(e)) or np.any(e < self.energy[0])
                or np.any(e > self.energy[-1])):
            raise ValueError("Energy outside tabulated range; no extrapolation")
        return np.exp(np.interp(np.log(e), np.log(self.energy), np.log(self.values)))


def number(x, name, low=0.0, strict=False):
    if isinstance(x, (bool, str)) or not np.isscalar(x):
        raise ValueError(f"{name} must be a number")
    x = float(x)
    if not np.isfinite(x) or (x <= low if strict else x < low):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {low}")
    return x


def keys(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(expected)}")


def validate(config):
    c = copy.deepcopy(config)
    keys(c, CONFIG, "config")
    for group in ("laser", "source"):
        keys(c[group], CONFIG[group], group)
    keys(SOURCE_MODEL,
         ("laser_to_hot_electron_fraction", "temperature_keV",
          "angular_sigma_rad"),
         "SOURCE_MODEL")
    for name, value in c["laser"].items():
        number(value, name, strict=True)
    number(c["laser"]["M_squared"], "M_squared", low=1)
    if c["target"] not in ("Al", "SS", "Teflon"):
        raise ValueError("target must be Al, SS or Teflon")
    if number(SOURCE_MODEL["laser_to_hot_electron_fraction"],
              "conversion fraction") > 1:
        raise ValueError("conversion fraction must not exceed 1")
    number(SOURCE_MODEL["angular_sigma_rad"], "angular sigma", strict=True)
    if number(c["source"]["observation_angle"],
              "observation angle") > np.pi/2:
        raise ValueError("observation angle must be between 0 and pi/2 radians")
    if SOURCE_MODEL["temperature_keV"] is not None:
        number(SOURCE_MODEL["temperature_keV"], "temperature", low=0.05)
    if set(DENSITY) != {"Al", "SS", "air", "BK7", "BaF2"}:
        raise ValueError("DENSITY must define Al, SS, air, BK7 and BaF2")
    for name, rho in DENSITY.items():
        number(rho, name+" density", strict=True)
    if not isinstance(c["path"], list) or not c["path"]:
        raise ValueError("path must be a nonempty list of ray segments")
    for p in c["path"]:
        keys(p, ("material", "thickness"), "path segment")
        if p["material"] not in ("vacuum", "air", "Al", "SS", "BK7", "BaF2"):
            raise ValueError("path materials must be vacuum, air, Al, SS, BK7 or BaF2")
        number(p["thickness"], "segment thickness")
    if sum(p["thickness"] for p in c["path"]) <= 0:
        raise ValueError("Total source-to-observer distance must be positive")
    return c


def laser_parameters(laser):
    w_lens = laser["beam_diameter"] / np.sqrt(2*np.log(2))
    w0 = (laser["M_squared"] * laser["wavelength"]
          * laser["focal_length"] / (np.pi*w_lens))
    tau_eff = laser["pulse_duration"] * np.sqrt(np.pi)/(2*np.sqrt(np.log(2)))
    intensity = laser["pulse_energy"] / (np.pi*w0*w0/2 * tau_eff)  # W/m^2
    # The empirical temperature law defines Q in W/cm^2 * um^2.
    q = intensity*laser["wavelength"]**2*1e8
    if not np.isfinite(q) or q <= 0:
        raise ValueError("Laser parameters outside numerical scope")
    if q <= 1.6e17:
        temperature = 6e-5*q**0.33
    else:
        u = q/1.37e18
        temperature = REST_KEV*u/(np.sqrt(1+u)+1)
    return w0, intensity, q, temperature


def angular_factor(sigma, theta):
    # Substitute theta=sigma*u, retaining the forward-hemisphere normalization.
    upper = min(np.pi/(2*sigma), 40.)
    integral = quad(lambda u: np.exp(-u*u/2)*np.sin(sigma*u)*sigma,
                    0, upper, epsabs=0, epsrel=1e-10)[0]
    if integral <= 0:
        raise ValueError("Angular normalization outside numerical scope")
    g = np.exp(-theta*theta/(2*sigma*sigma))/(2*np.pi*integral)
    if not np.isfinite(g) or g <= 0:
        raise ValueError("Angular factor underflows or overflows")
    return g


def yield_fraction(energy_keV, target):
    """Empirical thick-target yield; mixture yields are approximate surrogates."""
    weights = {"Al": {"Al": 1.0}, "SS": SS_WEIGHTS,
               "Teflon": TEFLON_WEIGHTS}[target]
    result = np.zeros_like(energy_keV, dtype=float)
    for element, weight in weights.items():
        a = 6e-4*ATOMIC_Z[element]*energy_keV/1000
        result += weight*a/(1+a)
    return result


def calculate(config, *, resolution=1, return_details=False):
    """Return H*(10), or dose and intermediate quantities when requested.

    resolution is an internal convergence-test multiplier, not a physical input.
    """
    c = validate(config)
    w0, intensity, q, t_auto = laser_parameters(c["laser"])
    T = (t_auto if SOURCE_MODEL["temperature_keV"] is None
         else SOURCE_MODEL["temperature_keV"])
    if not np.isfinite(T) or T < .05 or T > 1e5:
        raise ValueError("Temperature outside numerical scope (0.05-100000 keV)")
    hot_J = (c["laser"]["pulse_energy"]
             * SOURCE_MODEL["laser_to_hot_electron_fraction"])
    # NIST cm^2/g -> m^2/kg; dose coefficients pSv cm^2 -> Sv m^2.
    tables = {name: LogTable(data["rows"], 0.1)
              for name, data in NIST_TABLES.items()}
    hstar = LogTable(HSTAR_FLUENCE_TABLE, 1e-16)
    knots = np.concatenate([t.energy for t in [*tables.values(), hstar]])
    k = np.unique(np.r_[np.geomspace(K_MIN, K_MAX, int(18000*resolution)),
                         knots[(knots >= K_MIN) & (knots <= K_MAX)], .9*K_MAX])
    e = np.geomspace(1e-5, max(K_MAX, 80*T), int(90000*resolution))
    # Isotropic relativistic thermal number distribution in kinetic energy.
    f = (e+REST_KEV)*np.sqrt(e*(e+2*REST_KEV))*np.exp(-e/T)
    f /= trapezoid(f, e)
    mean = trapezoid(e*f, e)
    distance_m = sum(p["thickness"] for p in c["path"])
    g = angular_factor(SOURCE_MODEL["angular_sigma_rad"],
                       c["source"]["observation_angle"])
    details = {
        "focal_1_over_e2_radius": float(w0),
        "peak_intensity_W_m2": float(intensity),
        "Q_W_um2_per_cm2": float(q),
        "hot_electron_temperature_keV": float(T),
        "mean_electron_energy_keV": float(mean),
        "angular_factor_sr_inv": float(g),
        "source_to_observer_distance_m": float(distance_m),
    }
    if hot_J == 0:
        details["dose_mrem_per_pulse"] = 0.0
        return details if return_details else 0.0
    weighted = f*yield_fraction(e, c["target"])
    a = -cumulative_trapezoid(weighted[::-1], e[::-1], initial=0)[::-1]
    b = -cumulative_trapezoid((weighted/e)[::-1], e[::-1], initial=0)[::-1]
    # Escaping-source approximation: photon energy per pulse per keV, J/keV.
    spectrum = hot_J*2/mean*np.maximum(np.interp(k,e,a)-k*np.interp(k,e,b), 0)
    mu = {name: tables[name](k) for name in ("Al", "air")}
    mu["SS"] = sum(weight*tables[name](k) for name, weight in SS_WEIGHTS.items())
    mu["BK7"] = sum(weight*tables[name](k) for name, weight in BK7_WEIGHTS.items())
    mu["BaF2"] = sum(weight*tables[name](k) for name, weight in BAF2_WEIGHTS.items())
    optical_depth = np.zeros_like(k)
    for layer in c["path"]:
        name = layer["material"]
        if name != "vacuum":
            optical_depth += (mu[name]*DENSITY[name]
                              * layer["thickness"])
    # Photon fluence is photons/m^2/keV/pulse; hstar is Sv m^2/photon.
    phi = g*spectrum*np.exp(-optical_depth)/(distance_m**2*k*KEV_TO_J)
    dose_density_Sv = phi*hstar(k)
    dose_Sv = float(trapezoid(dose_density_Sv, k))
    if not np.isfinite(dose_Sv) or dose_Sv <= np.finfo(float).tiny:
        raise ValueError("Dose underflows or overflows; no resolved result")
    upper = k >= .9*K_MAX
    if trapezoid(dose_density_Sv[upper], k[upper])/dose_Sv > 1e-5:
        raise ValueError("Spectrum reaches the 20 MeV limit; wider data/model required")
    dose_mrem = dose_Sv*1e5  # 1 Sv = 1e5 mrem
    details["dose_mrem_per_pulse"] = dose_mrem
    return details if return_details else dose_mrem


def input_lines(config):
    """Return a deterministic, complete flattened listing of user inputs."""
    lines = []

    def visit(value, name):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}.{key}" if name else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]")
        else:
            lines.append(f"{name} = {json.dumps(value)}")

    visit(config, "")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="complete JSON configuration")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text()) if args.config else CONFIG
        result = calculate(config, return_details=True)
    except (ValueError, TypeError, KeyError, OSError, OverflowError, ZeroDivisionError) as exc:
        parser.error(str(exc))
    print("Input parameters:")
    print("\n".join(input_lines(config)))
    print("Calculated quantities:")
    print("focal_1_over_e2_radius = "
          f"{result['focal_1_over_e2_radius']:.8e}")
    print(f"peak_intensity_I0_W_m2 = {result['peak_intensity_W_m2']:.8e}")
    print(f"Q_W_um2_per_cm2 = {result['Q_W_um2_per_cm2']:.8e}")
    print("hot_electron_temperature_keV = "
          f"{result['hot_electron_temperature_keV']:.8e}")
    print("mean_electron_energy_keV = "
          f"{result['mean_electron_energy_keV']:.8e}")
    print(f"angular_factor_sr_inv = {result['angular_factor_sr_inv']:.8e}")
    print("source_to_observer_distance_m = "
          f"{result['source_to_observer_distance_m']:.8e}")
    print("Result:")
    print(f"H*(10) = {result['dose_mrem_per_pulse']:.8e} mrem/pulse")


if __name__ == "__main__":
    main()
