from enum import Enum, auto
from pystencils import Target


class Grid(Enum):
    UNIFORM = auto()
    STATIC_REFINEMENT = auto()
    DYNAMIC_REFINEMENT = auto()

    def is_uniform(self):
        return self in [Grid.UNIFORM]

    def is_refined(self):
        return self in [Grid.STATIC_REFINEMENT, Grid.DYNAMIC_REFINEMENT]

    @classmethod
    def to_enum(cls, grid_string):
        grid_string = grid_string.lower().replace('_', '')
        if grid_string == "uniform":
            return Grid.UNIFORM
        elif grid_string == "staticrefinement":
            return Grid.STATIC_REFINEMENT
        elif grid_string == "dynamicrefinement":
            raise ValueError("Dynamic refinement currently not supported.")
        else:
            raise ValueError("Invalid string for Grid Enum.")


class StreamingPattern(Enum):
    PULL = 'pull'
    PUSH = 'push'
    AA = 'aa'
    ESOTWIST = 'esotwist'
    ESOPULL = 'esopull'
    ESOPUSH = 'esopush'

    def is_inplace(self):
        return self in [StreamingPattern.AA, StreamingPattern.ESOTWIST,
                        StreamingPattern.ESOPULL, StreamingPattern.ESOPUSH]


class Fields(Enum):
    FLAG = 'flag'
    PDF = 'pdf'
    FORCE = 'force'
    DENSITY = 'density'
    VELOCITY = 'velocity'
    MEAN_VELOCITY = 'mean_velocity'
    SUM_OF_SQUARES = 'sum_of_squares'
    SUM_OF_CUBES = 'sum_of_cubes'
    OMEGA = 'omega'
    EDDY_VISCOSITY = 'eddy_viscosity'
    MEAN_EDDY_VISCOSITY = 'mean_eddy_viscosity'
    STRAIN_RATE = 'strain_rate'
    MEAN_STRAIN_RATE = 'mean_strain_rate'


class FieldType(Enum):
    PDF = auto()
    SCALAR = auto()
    VECTOR = auto()
    SECOND_ORDER_TENSOR = auto()
    THIRD_ORDER_TENSOR = auto()
    FLAG = auto()


def to_field_type(field: Fields):
    if field == Fields.PDF:
        return FieldType.PDF
    elif field == Fields.FLAG:
        return FieldType.FLAG
    elif field in [Fields.DENSITY, Fields.OMEGA, Fields.EDDY_VISCOSITY, Fields.MEAN_EDDY_VISCOSITY]:
        return FieldType.SCALAR
    elif field in [Fields.FORCE, Fields.VELOCITY, Fields.MEAN_VELOCITY]:
        return FieldType.VECTOR
    elif field in [Fields.SUM_OF_SQUARES, Fields.STRAIN_RATE, Fields.MEAN_STRAIN_RATE]:
        return FieldType.SECOND_ORDER_TENSOR
    elif field in [Fields.SUM_OF_CUBES]:
        return FieldType.THIRD_ORDER_TENSOR
    else:
        raise ValueError(f"Field type for field {field} is not defined.")


def source_extension(target):
    if (target == Target.CPU):
        return "cpp"
    else:
        return "cu"



def get_field_variable(field: Fields, target: Target = None):
    name = field.name.lower().replace("_", " ").title().replace(" ", "")
    return name[:1].lower() + name[1:] + f"Field{target.name.capitalize() if target else ''}ID"


def get_field_name(field: Fields, target: Target = None):
    return field.name.lower().replace("_", " ") + " field" + (" " + target.name.upper() if target else '')


def get_field_type(field: Fields, target: Target = None):
    type_name = to_field_type(field).name.lower().replace("_", " ").title().replace(" ", "")
    return type_name + (target.name.upper() if target else "") + "Field_T"


def device_to_host_field_copy(field: Fields):
    return (f"walberla::gpu::fieldCpy<{get_field_type(field, None)}, GPUField_T<real_t>>( "
            f"blocks, {get_field_variable(field, Target.CPU)}, {get_field_variable(field, Target.GPU)} );")
