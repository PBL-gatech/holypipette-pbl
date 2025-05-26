Pressure control
================

Holy Pipette can control different pressure controllers. All classes inherit
from `.PressureController` and expose the same ``set_pressure`` and ``measure``
API.

Supported controllers include:

- `.IBBPressureController`
- `.MoscowPressureController`
- `.FakePressureController` (development)

Examples
--------

Using the :class:`IBBPressureController`::

    from holypipette.devices.pressurecontroller import IBBPressureController
    import serial

    port = serial.Serial('COM5', 9600, timeout=0)
    controller = IBBPressureController(channel=1, arduinoSerial=port)
    controller.set_pressure(25)
    pressure = controller.measure()

The :class:`MoscowPressureController` uses separate ports for commands and
sensor readings::

    from holypipette.devices.pressurecontroller import MoscowPressureController
    import serial

    ctrl_serial = serial.Serial('COM5', 9600, timeout=0)
    read_serial = serial.Serial('COM9', 9600, timeout=0)
    controller = MoscowPressureController(channel=1,
                                          controllerSerial=ctrl_serial,
                                          readerSerial=read_serial)
    controller.set_pressure(25)
    pressure = controller.measure()

Pressure values are expressed in mBar.

Fake pressure controller
------------------------
For development purposes, a `.FakePressureController` is implemented.
It behaves as a pressure controller, except it is not connected to an
actual device.

