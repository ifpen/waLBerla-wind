
from dataclasses import replace

import pystencils as ps
{% if target == Target.GPU -%}
import numpy as np
{% endif -%}
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments, macroscopic_values_getter
from lbmpy.flow_statistics import welford_assignments
from lbmpy.utils import second_order_moment_tensor
from pystencils import Target
{% if target == Target.GPU -%}
from pystencils.fast_approximation import insert_fast_sqrts, insert_fast_divisions
from pystencils.typing import TypedSymbol
{% endif -%}
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.advanced_streaming import Timestep, get_timesteps
from lbmpy.advanced_streaming.utility import is_inplace
from lbmpy.boundaries import NoSlip, FreeSlip, UBB, ExtrapolationOutflow, WallFunctionBounce, MoninObukhovSimilarityTheory
from lbmpy_walberla.additional_data_handler import UBBAdditionalDataHandler
from lbmpy_walberla import generate_alternating_lbm_boundary
from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel
from lbmpy.turbulence_models import SubgridScaleModel

from pystencils_walberla import CodeGeneration, generate_info_header, generate_sweep {%- if grid == Grid.UNIFORM -%}, generate_pack_info_for_field {%- else -%}, generate_nonuniform_pack_info_for_field {%- endif %}
from lbmpy_walberla import generate_lbm_package, lbm_boundary_generator {%- if grid != Grid.UNIFORM -%}, RefinementScaling {%- endif %}
from lbmpy_walberla.additional_data_handler import OutflowAdditionalDataHandler

from lbmpy.relaxationrates import *

from codegen_walberla_wind import generate_flow_driver_collection
from codegen_walberla_wind.wall_function import WallLinkWFBAdditionalDataHandler

info_header = """

using flag_t = walberla::uint8_t;
using FlagField_T = walberla::FlagField<flag_t>;

using StorageSpecification_T = walberla::lbm::{name}_StorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;

using PdfField_T           = walberla::lbm_generated::PdfField< StorageSpecification_T >;
{% if target == Target.GPU -%}
using PdfGPUField_T        = walberla::lbm_generated::GPUPdfField< StorageSpecification_T >;
{% endif -%}
using BoundaryCollection_T = walberla::lbm::{name}_BoundaryCollection< FlagField_T >;

using SweepCollection_T = walberla::lbm::{name}_SweepCollection;

static const walberla::FlagUID FluidFlagUID("Fluid Flag");
static const walberla::FlagUID NoSlipFlagUID("{noslip_flag}");
static const walberla::FlagUID WFBFlagUID("{wfb_flag}");
static const walberla::FlagUID SymmetryFlagUID("{symmetry_flag}");
static const walberla::FlagUID UniformInflowFlagUID("{uniform_inflow_flag}");
static const walberla::FlagUID LogLawInflowFlagUID("{loglaw_inflow_flag}");
static const walberla::FlagUID OutflowFlagUID("{outflow_flag}");

namespace codegen {% raw %}{{{% endraw %}

    struct KernelInfo {% raw %}{{{% endraw %}

        static constexpr char stencil[]     = "{stencil}";
        static constexpr char method[]      = "{method}";
        static constexpr char forceModel[]  = "{forceModel}";
        
        static constexpr walberla::uint_t q = {q};
        
        static constexpr walberla::field::Layout layout = walberla::field::{layout};
        static constexpr char streamingPattern[] = "{streaming_pattern}";
        static constexpr bool inplace = {inplace};
        
        static constexpr bool compressible = {compressible};
        static constexpr bool zeroCentered = {zeroCentered};
        static constexpr char subgridScaleModel[] = "{subgridScaleModel}";
        
        static constexpr char cpuVectoriseInfo[] = "{cpuVectoriseInfo}";
        static constexpr char lbmOptimisationDict[] = "{lbmOptimisation}";
    
    {% raw %}}}{% endraw %};
    
    constexpr char KernelInfo::stencil[];
    constexpr char KernelInfo::method[];
    constexpr char KernelInfo::forceModel[];
    
    constexpr char KernelInfo::streamingPattern[];

    constexpr char KernelInfo::cpuVectoriseInfo[];
    constexpr char KernelInfo::lbmOptimisationDict[];
    
{% raw %}}}{% endraw %} // namespace codegen

"""

{%- if target == Target.GPU %}
compile_time_block_size = {{not runtime_gpu_indexing}}
max_threads = 256

if compile_time_block_size:
    sweep_block_size = (128, 1, 1)
else:
    sweep_block_size = (TypedSymbol("gpuBlockSize0", np.int32),
                        TypedSymbol("gpuBlockSize1", np.int32),
                        TypedSymbol("gpuBlockSize2", np.int32))

gpu_indexing_params = {'block_size': sweep_block_size}
{%- endif %}

{% if grid != Grid.UNIFORM %}ghost_layers = 2{% else %}ghost_layers = 1{%- endif %}

with CodeGeneration() as ctx:

    target = {{target}}
    layout = 'fzyx'
    streaming_pattern = '{{streaming_pattern.name.lower()}}'

    omega = sp.Symbol("omega")

    data_type = 'double' if ctx.double_accuracy else 'float32'

    density_field = ps.fields(f"density: {data_type}[3D]", layout=layout)
    velocity_field = ps.fields(f"velocity(3): {data_type}[3D]", layout=layout)
    {% if Fields.MEAN_VELOCITY in field_map -%}
    mean_velocity_field = ps.fields(f"mean_velocity(3): {data_type}[3D]", layout=layout)
    {% endif -%}
    {% if Fields.SUM_OF_SQUARES in field_map -%}
    sum_of_squares_field = ps.fields(f"sum_of_squares(9): {data_type}[3D]", layout=layout)
    {% endif -%}
    {% if Fields.SUM_OF_CUBES in field_map -%}
    sum_of_cubes_field = ps.fields(f"sum_of_cubes(27): {data_type}[3D]", layout=layout)
    {% endif -%}
    force_field = ps.fields(f"force(3): {data_type}[3D]", layout=layout)
    {% if Fields.OMEGA in field_map -%}
    omega_field = ps.fields(f"omega_out: {data_type}[3D]", layout=layout)
    {% endif -%}
    {% if Fields.EDDY_VISCOSITY in field_map -%}
    eddy_viscosity_field = ps.fields(f"eddy_viscosity: {data_type}[3D]", layout=layout)
    {% endif -%}
    {% if Fields.MEAN_EDDY_VISCOSITY in field_map -%}
    mean_eddy_viscosity_field = ps.fields(f"mean_eddy_viscosity: {data_type}[3D]", layout=layout)
    {% endif -%}
    {% if Fields.STRAIN_RATE in field_map -%}
    strain_rate_field = ps.fields(f"strain_rate(9): {data_type}[3D]", layout=layout)
    {% endif %}
    {% if Fields.MEAN_STRAIN_RATE in field_map -%}
    mean_strain_rate_field = ps.fields(f"mean_strain_rate(9): {data_type}[3D]", layout=layout)
    {% endif %}

    # lattice Boltzmann method

    stencil = LBStencil({{stencil}})
    q = stencil.Q

    pdfs, pdfs_tmp = ps.fields(f"pdfs({q}), pdfs_tmp({q}): {data_type}[3D]", layout=layout)
    macroscopic_fields = {'density': density_field, 'velocity': velocity_field}

    lbm_config = LBMConfig(stencil=stencil, streaming_pattern=streaming_pattern,
                           method=Method.CUMULANT, relaxation_rate=omega,
                           {% if stencil == Stencil.D3Q27 -%}galilean_correction=True, fourth_order_correction=0.1,{%- endif %}
                           force_model=ForceModel.GUO, force=force_field.center_vector,
                           subgrid_scale_model=SubgridScaleModel.QR,
                           compressible=True, zero_centered=True,
                           {% if Fields.OMEGA in field_map -%}
                           omega_output_field=omega_field,
                           {% endif -%}
                           {% if Fields.EDDY_VISCOSITY in field_map -%}
                           eddy_viscosity_field=eddy_viscosity_field,
                           {% endif -%}
                           output=macroscopic_fields)

    lbm_optimisation = LBMOptimisation(field_layout=layout, symbolic_field=pdfs, cse_global=True, cse_pdfs=False)

    if not is_inplace(streaming_pattern):
        lbm_opt = replace(lbm_optimisation, symbolic_temporary_field=pdfs_tmp)
        field_swaps = [(pdfs, pdfs_tmp)]
    else:
        field_swaps = []

    collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_optimisation)
    {% if target == Target.GPU -%}
    collision_rule = insert_fast_divisions(collision_rule)
    collision_rule = insert_fast_sqrts(collision_rule)
    {%- endif %}

    lb_method = collision_rule.method

    {% if vtk_output -%}
    # Welford update for output
    {% if Fields.MEAN_VELOCITY in field_map -%}
    {% if Fields.SUM_OF_CUBES in field_map -%}
    welford_output_update = welford_assignments(field=velocity_field, mean_field=mean_velocity_field,
                                                sum_of_squares_field=sum_of_squares_field,
                                                sum_of_cubes_field=sum_of_cubes_field)
    {% elif Fields.SUM_OF_SQUARES in field_map -%}
    welford_output_update = welford_assignments(field=velocity_field, mean_field=mean_velocity_field,
                                                sum_of_squares_field=sum_of_squares_field)
    {% else -%}
    welford_output_update = welford_assignments(field=velocity_field, mean_field=mean_velocity_field)
    {% endif -%}
    generate_sweep(ctx, "{{"waLBerlaWind_" + welford_output_name}}", welford_output_update, target=target{%- if target == Target.GPU %},
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})
    {%- endif %}

    {% if vtk_output and Fields.SUM_OF_SQUARES in field_map %}
    @ps.kernel
    def sos_resetter():
        for d in range(stencil.D**2):
            field_access = sum_of_squares_field.center.at_index(d)
            field_access @= sp.Float(0)
    generate_sweep(ctx, "waLBerlaWind_SoSResetter", ps.AssignmentCollection(sos_resetter), target=target{%- if target == Target.GPU %},
    gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})
    {% endif %}
    {% if vtk_output and Fields.SUM_OF_CUBES in field_map %}
    @ps.kernel
    def soc_resetter():
        for d in range(stencil.D**3):
            field_access = sum_of_cubes_field.center.at_index(d)
            field_access @= sp.Float(0)
    generate_sweep(ctx, "waLBerlaWind_SoCResetter", ps.AssignmentCollection(soc_resetter), target=target{%- if target == Target.GPU %},
    gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})
    {% endif %}

    {% if Fields.MEAN_EDDY_VISCOSITY in field_map -%}
    welford_nut_update = welford_assignments(field=eddy_viscosity_field, mean_field=mean_eddy_viscosity_field)
    generate_sweep(ctx, "{{"waLBerlaWind_" + welford_eddy_viscosity_name}}", welford_nut_update, target=target{%- if target == Target.GPU %},
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})
    {%- endif %}
    {% if Fields.MEAN_STRAIN_RATE in field_map -%}
    welford_strain_update = welford_assignments(field=strain_rate_field, mean_field=mean_strain_rate_field)
    generate_sweep(ctx, "{{"waLBerlaWind_" + welford_strain_rate_name}}", welford_strain_update, target=target{%- if target == Target.GPU %},
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})
    {%- endif %}
    {% endif %}

    #   PDF Setter -> used for initialisation before 0th timestep
    initial_rho = sp.Symbol('rho_0')
    pdfs_setter = pdf_initialization_assignments(lb_method=lb_method,
                                                 density=density_field.center,
                                                 velocity=velocity_field.center_vector,
                                                 pdfs=pdfs,
                                                 streaming_pattern=streaming_pattern, previous_timestep=get_timesteps(streaming_pattern)[0])

    generate_sweep(ctx, "waLBerlaWind_PdfSetter", pdfs_setter, ghost_layers_to_include=ghost_layers)

    #   Macro getter -> used when loading snapshot
    {% if streaming_pattern.is_inplace() -%}
    getter_assignments_even = macroscopic_values_getter(lb_method, velocity=velocity_field.center_vector,
                                                        pdfs=pdfs, density=density_field.center,
                                                        streaming_pattern=streaming_pattern, previous_timestep=Timestep.EVEN)
    generate_sweep(ctx, 'waLBerlaWind_MacroGetter_EVEN', getter_assignments_even)

    getter_assignments_odd = macroscopic_values_getter(lb_method, velocity=velocity_field.center_vector,
                                                       pdfs=pdfs, density=density_field.center,
                                                       streaming_pattern=streaming_pattern, previous_timestep=Timestep.ODD)
    generate_sweep(ctx, 'waLBerlaWind_MacroGetter_ODD', getter_assignments_odd)
    {%- else -%}
    getter_assignments = macroscopic_values_getter(lb_method, velocity=velocity_field.center_vector,
                                                   pdfs=pdfs.center_vector, density=density_field.center)
    generate_sweep(ctx, 'waLBerlaWind_MacroGetter', getter_assignments)
    {%- endif %}

    # MACROSCOPIC FIELD PACK INFOS
    {% if grid == Grid.UNIFORM -%}
    generate_pack_info_for_field(ctx, 'waLBerlaWind_DensityPackInfo', density_field, target=target)
    generate_pack_info_for_field(ctx, 'waLBerlaWind_VelocityPackInfo', velocity_field, target=target)
    {%- else -%}
    generate_nonuniform_pack_info_for_field(ctx, 'waLBerlaWind_DensityPackInfo', density_field, target=target)
    generate_nonuniform_pack_info_for_field(ctx, 'waLBerlaWind_VelocityPackInfo', velocity_field, target=target)
    {%- endif %}

    # GENERATE BOUNDARIES
    noslip_uid = 'NoSlip Flag'
    wfb_uid = 'WFB Flag'
    symmetry_uid = 'Symmetry Flag'
    uniform_inflow_uid = 'Uniform Inflow Flag'
    loglaw_inflow_uid = 'LogLaw Inflow Flag'
    outflow_uid = 'Outflow Flag'

    free_slip = lbm_boundary_generator(class_name='waLBerlaWind_FreeSlip', flag_uid=symmetry_uid,
                                       boundary_object=FreeSlip(lb_method.stencil, normal_direction=(0, 0, -1)))

    no_slip = lbm_boundary_generator(class_name='waLBerlaWind_NoSlip', flag_uid=noslip_uid, boundary_object=NoSlip())
    {% if generate_wfb -%}
    filter_width = sp.Symbol("velocity_filter_width")
    wfb_bc = WallFunctionBounce(lb_method=lb_method, normal_direction=(0, 0, 1),
                                reference_velocity=WallFunctionBounce.ReferenceVelocity.FILTERED_VELOCITY,
                                wall_function_model=MoninObukhovSimilarityTheory(sp.Symbol("z0")),
                                velocity_field=velocity_field,
                                sampling_shift=sp.Symbol("sampling_shift"), use_maronga_correction=True,
                                filter_width=filter_width,
                                data_type=data_type)
    wfb = lbm_boundary_generator(class_name='waLBerlaWind_WFB', flag_uid=wfb_uid,
                                 boundary_object=wfb_bc,
                                 additional_data_handler=WallLinkWFBAdditionalDataHandler(lb_method.stencil, wfb_bc,
                                                                                          velocity_field, target=target))
    {% endif %}

    uniform_ubb = lbm_boundary_generator(class_name='waLBerlaWind_UniformUBB', flag_uid=uniform_inflow_uid,
                                 boundary_object=UBB(velocity=[sp.Symbol("u_x"), sp.Symbol("u_y"), sp.Symbol("u_z")],
                                                     data_type=data_type))

    loglaw_ubb = lbm_boundary_generator(class_name='waLBerlaWind_LogLawUBB', flag_uid=loglaw_inflow_uid,
                                 boundary_object=UBB(lambda *args: None, dim=stencil.D, data_type=data_type))

    outflow_bc = ExtrapolationOutflow(normal_direction=(1, 0, 0),
                                      lb_method=lb_method, data_type=data_type,
                                      streaming_pattern=streaming_pattern, zeroth_timestep=Timestep.EVEN)
    outflow = lbm_boundary_generator(class_name='waLBerlaWind_Outflow', flag_uid=outflow_uid,
                                     boundary_object=outflow_bc,
                                     additional_data_handler=OutflowAdditionalDataHandler(lb_method.stencil, outflow_bc, target=target))

    {% if vtk_output -%}
    {% if Fields.STRAIN_RATE in field_map %}
    # strain rate writer
    @ps.kernel
    def strain_rate_writer():
        f_neq = sp.Matrix(pdfs.center_vector) - lb_method.get_equilibrium_terms()
        rho = lb_method.conserved_quantity_computation.density_symbol
        strain_rate_field.center_vector @= - 3 * omega_field.center / (2 * rho) * second_order_moment_tensor(f_neq, lb_method.stencil)

    strain_rate_ac = ps.AssignmentCollection(
        [lb_method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(pdfs.center_vector),
         *strain_rate_writer]
    )

    generate_sweep(ctx, "waLBerlaWind_StrainRateWriter", strain_rate_ac, target=target{%- if target == Target.GPU %},
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})
    {% endif -%}
    {% endif -%}

    {# take care of refinement scaling #}

    ### FLOW DRIVERS
    generate_flow_driver_collection(ctx, "FlowDriverCollection", force_field=force_field, velocity_field=velocity_field,
                                    nonuniform={% if grid == Grid.UNIFORM -%}False{%- else %}True{%- endif %},
                                    target=target, ghost_layers_to_include=ghost_layers, namespace='wind'{%- if target == Target.GPU %},
                                    gpu_indexing_params=gpu_indexing_params, max_threads=max_threads{%- endif %})

    name = 'waLBerlaWind'
    cpu_vectorise_info = {'nontemporal': True}
    generate_lbm_package(ctx, name=f"{name}_",
                         collision_rule=collision_rule,
                         lbm_config=lbm_config, lbm_optimisation=lbm_optimisation,
                         nonuniform={% if grid == Grid.UNIFORM -%}False{%- else %}True{%- endif %}, boundaries=[no_slip, free_slip, {% if generate_wfb -%}wfb, {% endif %}uniform_ubb, loglaw_ubb, outflow],
                         macroscopic_fields=macroscopic_fields,
                         target=target, {%- if target == Target.GPU %} gpu_indexing_params=gpu_indexing_params, max_threads=max_threads,{%- endif %}
                         cpu_vectorize_info=cpu_vectorise_info)

    info_header_params = {
        'name': name,
        'stencil': lbm_config.stencil.name,
        'q': q,
        'method': type(lbm_config.method).__name__,
        'forceModel': type(lbm_config.force_model).__name__,
        'layout': layout,
        'streaming_pattern': streaming_pattern,
        'inplace': 'true' if is_inplace(streaming_pattern) else 'false',
        'compressible': 'true' if lbm_config.compressible else 'false',
        'zeroCentered': 'true' if lbm_config.zero_centered else 'false',
        'subgridScaleModel': lbm_config.subgrid_scale_model if lbm_config.subgrid_scale_model else 'false',
        'cpuVectoriseInfo': str(cpu_vectorise_info),
        'lbmOptimisation': str(vars(lbm_optimisation)),
        'noslip_flag': noslip_uid,
        'wfb_flag': wfb_uid,
        'symmetry_flag': symmetry_uid,
        'uniform_inflow_flag': uniform_inflow_uid,
        'loglaw_inflow_flag': loglaw_inflow_uid,
        'outflow_flag': outflow_uid
    }

    field_typedefs = {
        'ScalarField_T': ps.fields(f"dummy: {data_type}[3D]", layout=layout),
        'VectorField_T': ps.fields(f"dummy(3): {data_type}[3D]", layout=layout),
        'SecondOrderTensorField_T': ps.fields(f"dummy(9): {data_type}[3D]", layout=layout),
        'ThirdOrderTensorField_T': ps.fields(f"dummy(27): {data_type}[3D]", layout=layout)
    }

    {{target|set_additional_headers(grid)|indent(4)}}

    generate_info_header(ctx, 'waLBerlaWind_KernelInfo',
                         field_typedefs=field_typedefs,
                         additional_headers=additional_headers,
                         additional_code=info_header.format(**info_header_params)
                         )
