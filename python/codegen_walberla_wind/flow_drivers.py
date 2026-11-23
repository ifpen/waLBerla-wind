import abc
from collections import OrderedDict
from enum import IntEnum
from typing import OrderedDict

import pystencils as ps
import sympy as sp

class AbstractFlowDriver(abc.ABC):
    class Direction(IntEnum):
        X = 0,
        Y = 1,
        Z = 2

    def __init__(self, force_field: [ps.Field, ps.Field.Access],
                 flow_direction: Direction, vertical_direction: Direction,
                 scaling_factor: [float, sp.Float, sp.Symbol]):
        if isinstance(force_field, ps.Field):
            self._force_field = force_field.center
        elif isinstance(force_field, ps.Field.Access):
            self._force_field = force_field
        else:
            raise TypeError("force_field must be a Field or Field.Access")

        self._dim = self._force_field.field.values_per_cell()
        self._flow_direction = flow_direction.value
        self._vertical_direction = vertical_direction.value
        self._lateral_direction = self._dim - self._flow_direction - self._vertical_direction

        self._scaling_factor = scaling_factor

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Flow driver class has to overwrite __call__")

    @property
    def dim(self):
        return self._dim

    @property
    def force_field(self):
        return self._force_field

class ZeroForce(AbstractFlowDriver):
    def __init__(self, force_field: [ps.Field, ps.Field.Access], scaling_factor, **kwargs):
        super(ZeroForce, self).__init__(force_field,
                                        AbstractFlowDriver.Direction.X,
                                        AbstractFlowDriver.Direction.Z,
                                        scaling_factor)

    def __call__(self, *args, **kwargs):
        assignments = list()
        for d in range(self._dim):
            assignments.append(ps.Assignment(self._force_field.at_index(d), sp.Float(0)))
        return assignments

class PressureGradient(AbstractFlowDriver):
    def __init__(self, force_field: [ps.Field, ps.Field.Access], scaling_factor,
                 flow_direction: AbstractFlowDriver.Direction = AbstractFlowDriver.Direction.X,
                 vertical_direction: AbstractFlowDriver.Direction = AbstractFlowDriver.Direction.Z,
                 **kwargs):
        super(PressureGradient, self).__init__(force_field, flow_direction, vertical_direction, scaling_factor)
        self._pressure_gradient = sp.symbols(f"pressureGradient")
        # cos/sin of the wind direction are kernel parameters, computed once on the host before the kernel launch
        self._cos_wind_direction = sp.Symbol("cosWindDirection")
        self._sin_wind_direction = sp.Symbol("sinWindDirection")

    def __call__(self, *args, **kwargs):

        x = self._flow_direction
        y = self._lateral_direction
        z = self._vertical_direction

        assignments = list()

        assignments.append(ps.Assignment(self._force_field.at_index(x), self._cos_wind_direction * self._pressure_gradient * self._scaling_factor))
        assignments.append(ps.Assignment(self._force_field.at_index(y), self._sin_wind_direction * self._pressure_gradient * self._scaling_factor))
        assignments.append(ps.Assignment(self._force_field.at_index(z), sp.Float(0)))

        return assignments

    @property
    def pressure_gradient(self):
        return self._pressure_gradient

    @property
    def cos_wind_direction(self):
        return self._cos_wind_direction

    @property
    def sin_wind_direction(self):
        return self._sin_wind_direction


class DynamicPressureGradient(PressureGradient):
    def __init__(self, force_field: [ps.Field, ps.Field.Access], velocity_field: [ps.Field, ps.Field.Access],
                 scaling_factor,
                 flow_direction: AbstractFlowDriver.Direction = AbstractFlowDriver.Direction.X,
                 vertical_direction: AbstractFlowDriver.Direction = AbstractFlowDriver.Direction.Z,
                 **kwargs):
        super(DynamicPressureGradient, self).__init__(force_field, scaling_factor, flow_direction, vertical_direction)

        self._wind_direction_rate = sp.Symbol( "windDirectionRate")

        if isinstance(velocity_field, ps.Field):
            self._velocity_field = velocity_field.center
        elif isinstance(force_field, ps.Field.Access):
            self._velocity_field = velocity_field
        else:
            raise TypeError("velocity_field must be a Field or Field.Access")


    def __call__(self, *args, **kwargs):
        x = self._flow_direction
        y = self._lateral_direction
        z = self._vertical_direction

        assignments = list()

        assignments.append(ps.Assignment(
            self._force_field.at_index(x),
            (self._cos_wind_direction * self._pressure_gradient
            - sp.Rational(1, 2) * self._wind_direction_rate * self._velocity_field.at_index(y)) * self._scaling_factor))
        assignments.append(ps.Assignment(
            self._force_field.at_index(y),
            (self._sin_wind_direction * self._pressure_gradient
            + sp.Rational(1, 2) * self._wind_direction_rate * self._velocity_field.at_index(x)) * self._scaling_factor))
        assignments.append(ps.Assignment(self._force_field.at_index(z), sp.Float(0)))

        return assignments

    @property
    def wind_direction_rate(self):
        return self._wind_direction_rate


class CoriolisForce(AbstractFlowDriver):
    def __init__(self, force_field: [ps.Field, ps.Field.Access], velocity_field: [ps.Field, ps.Field.Access],
                 scaling_factor,
                 flow_direction: AbstractFlowDriver.Direction = AbstractFlowDriver.Direction.X,
                 vertical_direction: AbstractFlowDriver.Direction = AbstractFlowDriver.Direction.Z,
                 **kwargs):
        super(CoriolisForce, self).__init__(force_field, flow_direction, vertical_direction, scaling_factor=scaling_factor)

        assert self._dim == 3, "Coriolis force is only defined for three-dimensional simulations."

        if isinstance(velocity_field, ps.Field):
            self._velocity_field = velocity_field.center
        elif isinstance(force_field, ps.Field.Access):
            self._velocity_field = velocity_field
        else:
            raise TypeError("velocity_field must be a Field or Field.Access")

        self._coriolis_frequency = sp.Symbol("coriolisFrequency")
        self._geostrophic_wind = sp.symbols(f"geostrophicWind_x:z")

    def __call__(self, *args, **kwargs):
        assignments = list()
        x = self._flow_direction
        y = self._lateral_direction
        z = self._vertical_direction

        assignments.append(ps.Assignment(
            self._force_field.at_index(x),
            -self._coriolis_frequency * (self._geostrophic_wind[y] - self._velocity_field.at_index(y)) * self._scaling_factor))
        assignments.append(ps.Assignment(
            self._force_field.at_index(y),
            +self._coriolis_frequency * (self._geostrophic_wind[x] - self._velocity_field.at_index(x)) * self._scaling_factor))
        assignments.append(ps.Assignment(self._force_field.at_index(z), sp.Float(0)))

        return assignments

    @property
    def coriolis_frequency(self):
        return self._coriolis_frequency

    @property
    def geostrophic_wind(self):
        return self._geostrophic_wind

    @property
    def vertical_direction(self):
        return self._vertical_direction


driver_dict = OrderedDict()
driver_dict['None'] = ZeroForce
driver_dict['PressureGradient'] = PressureGradient
driver_dict['DynamicPressureGradient'] = DynamicPressureGradient
driver_dict['CoriolisForce'] = CoriolisForce
