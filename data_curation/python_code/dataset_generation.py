import numpy as np
from scipy.interpolate import interp1d

# Constants
hbc = 197.4  # hbar*c (eV*nm)
LED_Min_lam = 400
LED_Max_lam = 1000
theta_t = np.pi/2
# Create lambda and w vectors
lambda_vec = np.linspace(LED_Min_lam, LED_Max_lam, LED_Max_lam - LED_Min_lam + 1)  # in nm
w_vec = 1240. / lambda_vec  # w in eV, lambda in nm

# Sample parameters (these should be defined as needed)
hval = np.arange(1, 11)  # h values
R1 = np.arange(5, 26)  # R1 values
ST = np.arange(1, 11)  # ST values
theta_list = np.arange(0, 27)  # theta values in degrees

def functionn(isHex, d, w_vec, eps_met, eps0, eps1, eps2, eps4, ft, n, eps_SiO2, ii, mm, hval, R1, ST, theta_list):
    ST_len = len(ST)
    theta_list_len = len(theta_list)

    Rs = np.zeros((ST_len, theta_list_len), dtype=float)
    Rp = np.zeros((ST_len, theta_list_len), dtype=float)
    Ts = np.zeros((ST_len, theta_list_len), dtype=float)
    Tp = np.zeros((ST_len, theta_list_len), dtype=float)

    for nn in range(ST_len):
        for tt in range(theta_list_len):
            hs = hval[ii]
            r1 = R1[mm]
            st = ST[nn]
            g = 0
            a = 2 * (r1 + st) + g
            theta_deg = theta_list[tt]

            theta = np.radians(theta_deg)
            h = r1 + st + hs
            k0 = (w_vec / hbc) * np.sqrt(eps0) * np.cos(theta)
            k1 = (w_vec / hbc) * np.sqrt(eps1 - eps0 * np.sin(theta)**2)
            k2 = (w_vec / hbc) * np.sqrt(eps2 - eps0 * np.sin(theta)**2)
            k4 = (w_vec / hbc) * np.sqrt(eps4 - eps0 * np.sin(theta)**2)

            e1 = eps_met
            e2 = eps_SiO2
            r2 = r1 + st

            P = 1 - (r1 / r2)**3
            ea = e1 * (3 - 2 * P) + 2 * e2 * P
            eb = e1 * P + e2 * (3 - P)

            chi_l = eps4 * (r2**3) * (e2 * ea - eps4 * eb) / (e2 * ea + 2 * eps4 * eb)

            if isHex == 1:
                Ua = 11.03354  # for hexagonal lattice
                beta_p = chi_l / (1 + chi_l * (1 / eps4) * (-0.5 * Ua / a**3))  # parallel
                beta_o = chi_l / (1 + chi_l * (1 / eps4) * (Ua / a**3))  # perpendicular
                f = 1.732 / 2
            else:
                Ua = 9.031  # for square lattice
                beta_p = chi_l / (1 + chi_l * (1 / eps4) * (-0.5 * Ua / a**3))  # parallel
                beta_o = chi_l / (1 + chi_l * (1 / eps4) * (Ua / a**3))  # perpendicular
                f = 1

            eps3_p = eps4 + 4 * np.pi * beta_p / (f * d * a**2)
            eps3_o = eps4**2 / (eps4 - 4 * np.pi * beta_o / (f * d * a**2))

            k3_p = (w_vec / hbc) * np.sqrt(eps3_p - eps0 * np.sin(theta)**2)
            k3_o = (w_vec / hbc) * np.sqrt(eps3_p / eps3_o) * np.sqrt(eps3_o - eps0 * np.sin(theta)**2)

            # Calculate reflection and transmission coefficients
            s1n = np.exp(1j * k1 * ft)
            s2n = np.exp(1j * k2 * (h - d / 2))
            s3n_p = np.exp(1j * k3_p * d)
            s3n_o = np.exp(1j * k3_o * d)

            r01p = (eps0 * k1 - eps1 * k0) / (eps0 * k1 + eps1 * k0)
            r12p = (eps1 * k2 - eps2 * k1) / (eps1 * k2 + eps2 * k1)
            r23p = (eps2 * k3_o - eps3_p * k2) / (eps2 * k3_o + eps3_p * k2)
            r34p = (eps3_p * k4 - eps4 * k3_o) / (eps3_p * k4 + eps4 * k3_o)

            r01s = (k0 - k1) / (k0 + k1)
            r12s = (k1 - k2) / (k1 + k2)
            r23s = (k2 - k3_p) / (k2 + k3_p)
            r34s = (k3_p - k4) / (k3_p + k4)

            t01s = (2 * k0) / (k0 + k1)
            t12s = (2 * k1) / (k1 + k2)
            t23s = (2 * k2) / (k2 + k3_p)
            t34s = (2 * k3_p) / (k3_p + k4)

            t01p = (2 * np.sqrt(eps0) * np.sqrt(eps1) * k0) / (k0 * eps1 + k1 * eps0)
            t12p = (2 * np.sqrt(eps1) * np.sqrt(eps2) * k1) / (k1 * eps2 + k2 * eps1)
            t23p = (2 * np.sqrt(eps2) * np.sqrt(eps3_p) * k2) / (k2 * eps3_p + k3_o * eps2)
            t34p = (2 * np.sqrt(eps3_p) * np.sqrt(eps4) * k3_o) / (k3_o * eps4 + k4 * eps3_p)

            # Reflectance and Transmittance calculations
            rs = (((r12s * s1n) / s2n + r01s / (s1n * s2n)) / s3n_p + r34s * (s3n_p * (s1n * s2n + (r01s * r12s * s2n) / s1n) + r23s * s3n_p * ((r12s * s1n) / s2n + r01s / (s1n * s2n))) + (r23s * (s1n * s2n + (r01s * r12s * s2n) / s1n)) / s3n_p) / ((1 / (s1n * s2n) + (r01s * r12s * s1n) / s2n) / s3n_p + r34s * (s3n_p * ((r12s * s2n) / s1n + r01s * s1n * s2n) + r23s * s3n_p * (1 / (s1n * s2n) + (r01s * r12s * s1n) / s2n)) + (r23s * ((r12s * s2n) / s1n + r01s * s1n * s2n)) / s3n_p)

            rp = (((r12p * s1n) / s2n + r01p / (s1n * s2n)) / s3n_o + r34p * (s3n_o * (s1n * s2n + (r01p * r12p * s2n) / s1n) + r23p * s3n_o * ((r12p * s1n) / s2n + r01p / (s1n * s2n))) + (r23p * (s1n * s2n + (r01p * r12p * s2n) / s1n)) / s3n_o) / ((1 / (s1n * s2n) + (r01p * r12p * s1n) / s2n) / s3n_o + r34p * (s3n_o * ((r12p * s2n) / s1n + r01p * s1n * s2n) + r23p * s3n_o * (1 / (s1n * s2n) + (r01p * r12p * s1n) / s2n)) + (r23p * ((r12p * s2n) / s1n + r01p * s1n * s2n)) / s3n_o)

            ts = (t01s * t12s * t23s * t34s) / ((1 / (s1n * s2n) + (r01s * r12s * s1n) / s2n) / s3n_p +
                r34s * (s3n_p * ((r12s * s2n) / s1n + r01s * s1n * s2n) + r23s * s3n_p * (1 / (s1n * s2n) +
                (r01s * r12s * s1n) / s2n)) + (r23s * ((r12s * s2n) / s1n + r01s * s1n * s2n)) / s3n_p)

            tp = (t01p * t12p * t23p * t34p) / ((1 / (s1n * s2n) + (r01p * r12p * s1n) / s2n) / s3n_o +
                r34p * (s3n_o * ((r12p * s2n) / s1n + r01p * s1n * s2n) + r23p * s3n_o * (1 / (s1n * s2n) +
                (r01p * r12p * s1n) / s2n)) + (r23p * ((r12p * s2n) / s1n + r01p * s1n * s2n)) / s3n_o)

            Rs[nn, tt] = 100 * abs(rs)**2
            Rp[nn, tt] = 100 * abs(rp)**2
            Ts[nn, tt] = 100 * abs(ts)**2 * (np.sqrt(eps4) / np.sqrt(eps0)) * np.cos(theta) / np.cos(theta_t)
            Tp[nn, tt] = 100 * abs(tp)**2 * (np.sqrt(eps4) / np.sqrt(eps0)) * np.cos(theta) / np.cos(theta_t)

    return Rs, Rp, Ts, Tp

