from patcherbot.utils.config import Config, NumberWithUnit, Boolean
import logging


class ProtocolConfig(Config):
    """
    Control of patch-clamp protocols.
    """
    voltage_protocol = Boolean(default=True, doc='Run the Voltage Protocol automatically')
    current_protocol = Boolean(default=True, doc='Run the Current Protocol automatically')
    holding_protocol = Boolean(default=False, doc='Run the Holding Protocol automatically')
    voltage_sweep_protocol = Boolean(default=False, doc='Run the Voltage Sweep Protocol automatically')

    custom_cclamp_protocol = Boolean(default=False, doc='Customize the protocol parameters')
    cclamp_step = NumberWithUnit(30, bounds=(0, 3000), doc='Step Current', unit='pA', magnitude=1)
    cclamp_start = NumberWithUnit(-500, bounds=(-30000, 0), doc='Start Current', unit='pA', magnitude=1)
    cclamp_end = NumberWithUnit(500, bounds=(0, 30000), doc='End Current', unit='pA', magnitude=1)
    cclamp_hold = NumberWithUnit(-20, bounds=(-200, 0), doc='Holding Current', unit='pA', magnitude=1)
    hclamp_duration = NumberWithUnit(30, bounds=(0, 600), doc='Holding Protocol Duration', unit='s')

    vclamp_start = NumberWithUnit(-110e-3, bounds=(-200e-3, 0), doc='Voltage-clamp sweep start', unit='mV', magnitude=1e-3)
    vclamp_step = NumberWithUnit(20e-3, bounds=(0, 100e-3), doc='Voltage-clamp sweep step', unit='mV', magnitude=1e-3)
    vclamp_end = NumberWithUnit(50e-3, bounds=(0, 200e-3), doc='Voltage-clamp sweep end', unit='mV', magnitude=1e-3)
    vclamp_hold = NumberWithUnit(-110e-3, bounds=(-200e-3, 0), doc='Voltage-clamp sweep holding potential', unit='mV', magnitude=1e-3)

    categories = [
        ('Protocols', ['voltage_protocol', 'current_protocol', 'holding_protocol', 'voltage_sweep_protocol']),
        ('Protocol Param', ['custom_cclamp_protocol', 'cclamp_step', 'cclamp_start', 'cclamp_end', 'cclamp_hold', 'vclamp_start', 'vclamp_end', 'vclamp_hold', 'hclamp_duration']),
    ]

    logging.info("ProtocolConfig initialized successfully.")
