

from holypipette.config import Config, NumberWithUnit, Number, Boolean
import logging

class PatchConfig(Config):
    '''
    Control of automatic patch clamp algorithm
    '''
    # Define parameters directly without try-except
    pressure_near = NumberWithUnit(45, bounds=(0, 100), doc='Pressure during approach', unit='mbar')
    pressure_sealing = NumberWithUnit(-20, bounds=(-100, 0), doc='Pressure for sealing', unit='mbar')
    pressure_ramp_increment = NumberWithUnit(-25, bounds=(-100, 0), doc='Pressure ramp increment', unit='mbar')
    pressure_ramp_max = NumberWithUnit(-300., bounds=(-1000, 0), doc='Pressure ramp maximum', unit='mbar')
    pressure_ramp_duration = NumberWithUnit(1.15, bounds=(0, 10), doc='Pressure ramp duration', unit='s')

    min_R = NumberWithUnit(2e6, bounds=(0, 1000e6), doc='Minimum normal resistance', unit='MΩ', magnitude=1e6)
    max_R = NumberWithUnit(25e6, bounds=(0, 1000e6), doc='Maximum normal resistance', unit='MΩ', magnitude=1e6)
    max_cell_R = NumberWithUnit(300e6, bounds=(0, 1000e6), doc='Maximum cell resistance', unit='MΩ', magnitude=1e6)
    cell_distance = NumberWithUnit(25, bounds=(0, 100), doc='Initial distance above target cell', unit='μm')
    max_distance = NumberWithUnit(30, bounds=(0, 100), doc='Maximum movement during approach', unit='μm')

    max_R_increase = NumberWithUnit(1e6, bounds=(0, 100e6), doc='Increase in resistance indicating obstruction', unit='MΩ', magnitude=1e6)
    cell_R_increase = Number(.15, bounds=(0, 1), doc='Proportional increase in resistance indicating cell presence')
    gigaseal_R = NumberWithUnit(1000e6, bounds=(100e6, 10000e6), doc='Gigaseal resistance', unit='MΩ', magnitude=1e6)

    seal_min_time = NumberWithUnit(15, bounds=(0, 60), doc='Minimum time for seal', unit='s')
    seal_deadline = NumberWithUnit(90., bounds=(0, 300), doc='Maximum time for seal formation', unit='s')

    Vramp_duration = NumberWithUnit(10., bounds=(0, 60), doc='Voltage ramp duration', unit='s')
    Vramp_amplitude = NumberWithUnit(-70e-3, bounds=(-200e-3, 0), doc='Holding Potential', unit='mV', magnitude=1e-3)

    zap = Boolean(False, doc='Zap the cell to break the seal')

    voltage_protocol = Boolean(default = True, doc='Run the Voltage Protocol automatically')
    current_protocol = Boolean(default = True, doc='Run the Current Protocol automatically')
    holding_protocol = Boolean(default = True, doc='Run the Holding Protocol automatically')

    custom_protocol = Boolean(default = False, doc='Customize the protocol parameters')
    cclamp_step = NumberWithUnit(10, bounds=(0, 20), doc='Step Current', unit='pA', magnitude=1)
    cclamp_start = NumberWithUnit(-50, bounds=(-300, -20), doc='Start Current', unit='pA', magnitude=1)
    cclamp_end = NumberWithUnit(50, bounds=(0, 300), doc='End Current', unit='pA', magnitude=1)

    categories = [
        ('Approach', ['min_R', 'max_R', 'pressure_near', 'cell_distance', 'max_distance', 'cell_R_increase']),
        ('Sealing', ['pressure_sealing', 'gigaseal_R', 'Vramp_duration', 'Vramp_amplitude', 'seal_min_time', 'seal_deadline']),
        ('Break-in', ['zap', 'pressure_ramp_increment', 'pressure_ramp_max', 'pressure_ramp_duration', 'max_cell_R']),
        ('Protocols', ['voltage_protocol', 'current_protocol', 'holding_protocol']),
        ('Current Clamp', ['custom_protocol', 'cclamp_step', 'cclamp_start', 'cclamp_end']),
    ]

    logging.info("PatchConfig initialized successfully.")