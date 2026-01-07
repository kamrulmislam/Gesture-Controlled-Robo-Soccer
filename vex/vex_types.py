# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
""" 
AIM WebSocket API - Types

This module defines various types and enums used in the AIM WebSocket API.
"""
from enum import Enum
from typing import Union
import time
class vexEnum:
    '''Base class for all enumerated types'''
    value = 0
    name = ""

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return self.value

class SoundType(str, Enum):
    DOORBELL       = "DOORBELL"
    TADA           = "TADA"
    FAIL           = "FAIL"
    SPARKLE        = "SPARKLE"
    FLOURISH       = "FLOURISH"
    FORWARD        = "FORWARD"
    REVERSE        = "REVERSE"
    RIGHT          = "RIGHT"
    LEFT           = "LEFT"
    BLINKER        = "BLINKER"
    CRASH          = "CRASH"
    BRAKES         = "BRAKES"
    HUAH           = "HUAH"
    PICKUP         = "PICKUP"
    #PLACE          = "PLACE"
    #KICK           = "KICK"
    CHEER          = "CHEER"
    SENSING        = "SENSING"
    DETECTED       = "DETECTED"
    OBSTACLE       = "OBSTACLE"
    LOOPING        = "LOOPING"
    COMPLETE       = "COMPLETE"
    PAUSE          = "PAUSE"
    RESUME         = "RESUME"
    SEND           = "SEND"
    RECEIVE        = "RECEIVE"
    #CHIRP          = "CHIRP"

    ACT_HAPPY      = "ACT_HAPPY"
    ACT_SAD        = "ACT_SAD"
    ACT_EXCITED    = "ACT_EXCITED"
    ACT_ANGRY      = "ACT_ANGRY"
    ACT_SILLY      = "ACT_SILLY"

class FontType(str, Enum):
    MONO20 = "MONO20"
    MONO24 = "MONO24"
    MONO30 = "MONO30"
    MONO36 = "MONO36"
    MONO40 = "MONO40"
    MONO60 = "MONO60"
    PROP20 = "PROP20"
    PROP24 = "PROP24"
    PROP30 = "PROP30"
    PROP36 = "PROP36"
    PROP40 = "PROP40"
    PROP60 = "PROP60"
    MONO15 = "MONO15"
    MONO12 = "MONO12"

class KickType(str, Enum):
   SOFT          = "kick_soft"
   MEDIUM        = "kick_medium"
   HARD          = "kick_hard"

class AxisType(Enum):
    """The defined units for inertial sensor axis."""
    X_AXIS        = 0
    Y_AXIS        = 1
    Z_AXIS        = 2

class TurnType(Enum):
    LEFT        = 0
    RIGHT       = 1
class OrientationType:
    '''The defined units for inertial sensor orientation.'''
    ROLL        = 0
    PITCH       = 1
    YAW         = 2
class AccelerationType:
    '''The defined units for inertial sensor acceleration.'''
    FORWARD     = 0
    RIGHTWARD   = 1
    DOWNWARD    = 2

class PercentUnits:
    '''The measurement units for percentage values.'''
    class PercentUnits(vexEnum):
        pass
    PERCENT = PercentUnits(0, "PERCENT")
    '''A percentage unit that represents a value from 0% to 100%'''
class RotationUnits:
    '''The measurement units for rotation values.'''
    class RotationUnits(vexEnum):
        pass
    DEG = RotationUnits(0, "DEG")
    '''A rotation unit that is measured in degrees.'''
    REV = RotationUnits(1, "REV")
    '''A rotation unit that is measured in revolutions.'''
    RAW = RotationUnits(99, "RAW")
    '''A rotation unit that is measured in raw data form.'''
class DriveVelocityUnits:
    '''The measurement units for drive velocity values.'''
    class DriveVelocityUnits(vexEnum):
        pass
    PERCENT = DriveVelocityUnits(0, "PCT")
    '''A velocity unit that is measured in percentage.'''
    MMPS = DriveVelocityUnits(1, "MMPS")
    '''A velocity unit that is measured in mm per second.'''
class TurnVelocityUnits:
    '''The measurement units for turn velocity values.'''
    class TurnVelocityUnits(vexEnum):
        pass
    PERCENT = TurnVelocityUnits(0, "PCT")
    '''A velocity unit that is measured in percentage.'''
    DPS = TurnVelocityUnits(1, "DPS")
    '''A velocity unit that is measured in degrees per second.'''

class TimeUnits:
    '''The measurement units for time values.'''
    class TimeUnits(vexEnum):
        pass
    SECONDS = TimeUnits(0, "SECONDS")
    '''A time unit that is measured in seconds.'''
    MSEC = TimeUnits(1, "MSEC")
    '''A time unit that is measured in milliseconds.'''
class DistanceUnits:
    '''The measurement units for distance values.'''
    class DistanceUnits(vexEnum):
        pass
    MM = DistanceUnits(0, "MM")
    '''A distance unit that is measured in millimeters.'''
    IN = DistanceUnits(1, "IN")
    '''A distance unit that is measured in inches.'''
    CM = DistanceUnits(2, "CM")
    '''A distance unit that is measured in centimeters.'''
class VoltageUnits:
    '''The measurement units for voltage values.'''
    class VoltageUnits(vexEnum):
        pass
    VOLT = VoltageUnits(0, "VOLT")
    '''A voltage unit that is measured in volts.'''
    MV = VoltageUnits(0, "mV")
    '''A voltage unit that is measured in millivolts.'''

# ----------------------------------------------------------
# globals
# ----------------------------------------------------------
PERCENT = PercentUnits.PERCENT
'''A percentage unit that represents a value from 0% to 100%'''
LEFT = TurnType.LEFT
'''A turn unit that is defined as left turning.'''
RIGHT = TurnType.LEFT
'''A turn unit that is defined as right turning.'''
DEGREES = RotationUnits.DEG
'''A rotation unit that is measured in degrees.'''
TURNS = RotationUnits.REV
'''A rotation unit that is measured in revolutions.'''
SECONDS = TimeUnits.SECONDS
'''A time unit that is measured in seconds.'''
MSEC = TimeUnits.MSEC
'''A time unit that is measured in milliseconds.'''
INCHES = DistanceUnits.IN
'''A distance unit that is measured in inches.'''
MM = DistanceUnits.MM
'''A distance unit that is measured in millimeters.'''
VOLT = VoltageUnits.VOLT
'''A voltage unit that is measured in volts.'''
MV = VoltageUnits.MV
'''A voltage unit that is measured in millivolts.'''
MMPS = DriveVelocityUnits.MMPS
'''units of mm per second'''
DPS = TurnVelocityUnits.DPS
'''units of degrees per second'''
OFF = False
'''used to turn off an LED'''

vexnumber = Union[int, float]
# drivetrain move functions take either DriveVelocity or percentage units
DriveVelocityPercentUnits = Union[DriveVelocityUnits.DriveVelocityUnits, PercentUnits.PercentUnits]
# drivetrain turn functions take either TurnVelocity or percentage units
TurnVelocityPercentUnits = Union[TurnVelocityUnits.TurnVelocityUnits, PercentUnits.PercentUnits]

class LightType(str, Enum):
   LED1      = "light1"
   LED2      = "light2"
   LED3      = "light3"
   LED4      = "light4"
   LED5      = "light5"
   LED6      = "light6"
   ALL_LEDS  = "all"

class Color:
    '''### Color class - create a new color

    This class is used to create instances of color objects

    #### Arguments:
        value : The color value, can be specified in various ways, see examples.

    #### Returns:
        An instance of the Color class

    #### Examples:
        # create blue using hex value\\
        c = Color(0x0000ff)\n
        # create blue using r, g, b values\\
        c = Color(0, 0, 255)\n
        # create blue using web string\\
        c = Color("#00F")\n
        # create blue using web string (alternate)\\
        c = Color("#0000FF")\n
        # create red using an existing object\\
        c = Color(Color.RED)
    '''
    class DefinedColor:
        def __init__(self, value, transparent=False):
            self.value = value
            self.transparent = transparent

    BLACK       = DefinedColor(0x000000)
    '''predefined Color black'''
    WHITE       = DefinedColor(0xFFFFFF)
    '''predefined Color white'''
    RED         = DefinedColor(0xFF0000)
    '''predefined Color red'''
    GREEN       = DefinedColor(0x00FF00)
    '''predefined Color green'''
    BLUE        = DefinedColor(0x001871)
    '''predefined Color blue'''
    YELLOW      = DefinedColor(0xFFFF00)
    '''predefined Color yellow'''
    ORANGE      = DefinedColor(0xFF8500)
    '''predefined Color orange'''
    PURPLE      = DefinedColor(0xFF00FF)
    '''predefined Color purple'''
    CYAN        = DefinedColor(0x00FFFF)
    '''predefined Color cyan'''
    TRANSPARENT = DefinedColor(0x000000, True)
    '''predefined Color transparent'''

    def __init__(self, *args):
        self.transparent = False
        if len(args) == 1 and isinstance(args[0], int):
            self.value: int = args[0]
        elif len(args) == 3 and all(isinstance(arg, int) for arg in args):
            self.value = ((args[0] & 0xFF) << 16) + ((args[1] & 0xFF) << 8) + (args[2] & 0xFF)
        else:
            raise TypeError("bad parameters")
    
    def set_rgb(self, *args):
        '''### change existing Color instance to new rgb value

        #### Arguments:
            value : The color value, can be specified in various ways, see examples.

        #### Returns:
            integer value representing the color

        #### Examples:
            # create a color that is red
            c = Color(0xFF0000)
            # change color to blue using single value
            c.rgb(0x0000FF)
            # change color to green using three values
            c.rgb(0, 255, 0)
        '''
        if len(args) == 1 and isinstance(args[0], int):
            self.value = args[0]
        if len(args) == 3 and all(isinstance(arg, int) for arg in args):
            self.value = ((args[0] & 0xFF) << 16) + ((args[1] & 0xFF) << 8) + (args[2] & 0xFF)
    
    # ----------------------------------------------------------

def sleep(duration: vexnumber, units=TimeUnits.MSEC):
    '''### delay the current thread for the provided number of seconds or milliseconds.

    #### Arguments:
        duration: The number of seconds or milliseconds to sleep for
        units:    The units of duration, optional, default is milliseconds

    #### Returns:
        None
    '''
    if units == TimeUnits.MSEC:
        time.sleep(duration / 1000)
    else:
        time.sleep(duration)

def wait(duration: vexnumber, units=TimeUnits.MSEC):
    '''### delay the current thread for the provided number of seconds or milliseconds.

    #### Arguments:
        duration: The number of seconds or milliseconds to sleep for
        units:    The units of duration, optional, default is milliseconds

    #### Returns:
        None
    '''
    if units == TimeUnits.MSEC:
        time.sleep(duration / 1000)
    else:
        time.sleep(duration)

class EmojiType:
    class EmojiType(vexEnum):
        pass
    EXCITED = EmojiType( 0, "EXCITED")
    CONFIDENT = EmojiType( 1, "CONFIDENT")
    SILLY = EmojiType( 2, "SILLY")
    AMAZED = EmojiType( 3, "AMAZED")
    STRONG = EmojiType( 4, "STRONG")
    THRILLED = EmojiType( 5, "THRILLED")
    HAPPY = EmojiType( 6, "HAPPY")
    PROUD = EmojiType( 7, "PROUD")
    LAUGHING = EmojiType( 8, "LAUGHING")
    OPTIMISTIC = EmojiType(9, "OPTIMISTIC")
    DETERMINED = EmojiType(10, "DETERMINED")
    AFFECTIONATE = EmojiType(11, "AFFECTIONATE")
    CALM = EmojiType(12, "CALM")
    QUIET = EmojiType(13, "QUIET")
    SHY = EmojiType(14, "SHY")
    CHEERFUL = EmojiType(15, "CHEERFUL")
    LOVED = EmojiType(16, "LOVED")
    SURPRISED = EmojiType(17, "SURPRISED")
    THINKING = EmojiType(18, "THINKING")
    TIRED = EmojiType(19, "TIRED")
    CONFUSED = EmojiType(20, "CONFUSED")
    BORED = EmojiType(21, "BORED")
    EMBARRASSED = EmojiType(22, "EMBARRASSED")
    WORRIED = EmojiType(23, "WORRIED")
    SAD = EmojiType(24, "SAD")
    SICK = EmojiType(25, "SICK")
    DISAPPOINTED = EmojiType(26, "DISAPPOINTED")
    NERVOUS = EmojiType(27, "NERVOUS")
    ANNOYED = EmojiType(30, "ANNOYED")
    STRESSED = EmojiType(31, "STRESSED")
    ANGRY = EmojiType(32, "ANGRY")
    FRUSTRATED = EmojiType(33, "FRUSTRATED")
    JEALOUS = EmojiType(34, "JEALOUS")
    SHOCKED = EmojiType(35, "SHOCKED")
    FEAR = EmojiType(36, "FEAR")
    DISGUST = EmojiType(37, "DISGUST")

Emoji = EmojiType

# ----------------------------------------------------------

class EmojiLookType:
    class EmojiLookType(vexEnum):
        pass
    LOOK_FORWARD = EmojiLookType( 0, "LOOK_FORWARD")
    LOOK_RIGHT = EmojiLookType( 1, "LOOK_RIGHT")
    LOOK_LEFT = EmojiLookType( 2, "LOOK_LEFT")

EmojiLook = EmojiLookType

class StackingType(Enum):
   STACKING_OFF           = 0
   STACKING_MOVE_RELATIVE = 1
   STACKING_MOVE_GLOBAL   = 2

class SensitivityType:
    class SensitivityType(vexEnum):
        pass
    LOW = SensitivityType( 0, "LOW")
    MEDIUM = SensitivityType( 1, "MEDIUM")
    HIGH = SensitivityType( 2, "HIGH")
