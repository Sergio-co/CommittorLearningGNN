#!/usr/bin/env python3

# boltzmann constant in J/K
# source: https://www.nist.gov/si-redefinition/kelvin-boltzmann-constant
boltzmann_constant_jk = 1.380649*1e-23
# 1 cal to joule
cal_to_joule = 4.184
# Avogadro constant
# source: https://www.nist.gov/pml/weights-and-measures/si-units-amount-substance
avogadro_constant = 6.02214076*1e23

# boltzmann in kJ/(mol*K)
boltzmann_constant_kjmolk = boltzmann_constant_jk * avogadro_constant * 1e-3
# boltzmann in kcal/(mol*K)
boltzmann_constant_kcalmolk = boltzmann_constant_kjmolk / cal_to_joule

# standard concentration in angstrom^3
standard_concentration = 1.0/(avogadro_constant/1e27)

if __name__ == '__main__':
    print(f'{"Boltzmann constant in kJ/(mol⋅K):":36s} {boltzmann_constant_kjmolk:20.18f} ({boltzmann_constant_kjmolk:20.15e})')
    print(f'{"Boltzmann constant in kcal/(mol⋅K):":36s} {boltzmann_constant_kcalmolk:20.18f} ({boltzmann_constant_kcalmolk:20.15e})')
    print(f'Standard concentration 1 M in Å^3: 1/{standard_concentration:20.18f}')
