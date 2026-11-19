
from dataclasses import replace

import pystencils as ps
import numpy as np
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments, macroscopic_values_getter
from lbmpy.flow_statistics import welford_assignments
from lbmpy.utils import second_order_moment_tensor
from pystencils import Target
from pystencils.fast_approximation import insert_fast_sqrts, insert_fast_divisions
from pystencils.typing import TypedSymbol
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.advanced_streaming import Timestep, get_timesteps
from lbmpy.advanced_streaming.utility import is_inplace
from lbmpy.boundaries import NoSlip, FreeSlip, UBB, ExtrapolationOutflow, WallFunctionBounce, MoninObukhovSimilarityTheory
from lbmpy_walberla.additional_data_handler import UBBAdditionalDataHandler
from lbmpy_walberla import generate_alternating_lbm_boundary
from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel
from lbmpy.turbulence_models import SubgridScaleModel

from pystencils_walberla import CodeGeneration, generate_info_header, generate_sweep, generate_pack_info_for_field
from lbmpy_walberla import generate_lbm_package, lbm_boundary_generator
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
using PdfGPUField_T        = walberla::lbm_generated::GPUPdfField< StorageSpecification_T >;
using BoundaryCollection_T = walberla::lbm::{name}_BoundaryCollection< FlagField_T >;

using SweepCollection_T = walberla::lbm::{name}_SweepCollection;

static const walberla::FlagUID FluidFlagUID("Fluid Flag");
static const walberla::FlagUID NoSlipFlagUID("{noslip_flag}");
static const walberla::FlagUID WFBFlagUID("{wfb_flag}");
static const walberla::FlagUID SymmetryFlagUID("{symmetry_flag}");
static const walberla::FlagUID UniformInflowFlagUID("{uniform_inflow_flag}");
static const walberla::FlagUID LogLawInflowFlagUID("{loglaw_inflow_flag}");
static const walberla::FlagUID OutflowFlagUID("{outflow_flag}");

namespace codegen {{

    struct KernelInfo {{

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
    
    }};
    
    constexpr char KernelInfo::stencil[];
    constexpr char KernelInfo::method[];
    constexpr char KernelInfo::forceModel[];
    
    constexpr char KernelInfo::streamingPattern[];

    constexpr char KernelInfo::cpuVectoriseInfo[];
    constexpr char KernelInfo::lbmOptimisationDict[];
    
}} // namespace codegen

"""
compile_time_block_size = True
max_threads = 256

if compile_time_block_size:
    sweep_block_size = (128, 1, 1)
else:
    sweep_block_size = (TypedSymbol("gpuBlockSize0", np.int32),
                        TypedSymbol("gpuBlockSize1", np.int32),
                        TypedSymbol("gpuBlockSize2", np.int32))

gpu_indexing_params = {'block_size': sweep_block_size}

ghost_layers = 1

with CodeGeneration() as ctx:

    target = Target.GPU
    layout = 'fzyx'
    streaming_pattern = 'pull'

    omega = sp.Symbol("omega")

    data_type = 'double' if ctx.double_accuracy else 'float32'

    density_field = ps.fields(f"density: {data_type}[3D]", layout=layout)
    velocity_field = ps.fields(f"velocity(3): {data_type}[3D]", layout=layout)
    mean_velocity_field = ps.fields(f"mean_velocity(3): {data_type}[3D]", layout=layout)
    sum_of_squares_field = ps.fields(f"sum_of_squares(9): {data_type}[3D]", layout=layout)
    sum_of_cubes_field = ps.fields(f"sum_of_cubes(27): {data_type}[3D]", layout=layout)
    force_field = ps.fields(f"force(3): {data_type}[3D]", layout=layout)
    omega_field = ps.fields(f"omega_out: {data_type}[3D]", layout=layout)
    eddy_viscosity_field = ps.fields(f"eddy_viscosity: {data_type}[3D]", layout=layout)
    mean_eddy_viscosity_field = ps.fields(f"mean_eddy_viscosity: {data_type}[3D]", layout=layout)
    strain_rate_field = ps.fields(f"strain_rate(9): {data_type}[3D]", layout=layout)
    
    mean_strain_rate_field = ps.fields(f"mean_strain_rate(9): {data_type}[3D]", layout=layout)
    

    # lattice Boltzmann method

    stencil = LBStencil(Stencil.D3Q27)
    q = stencil.Q

    pdfs, pdfs_tmp = ps.fields(f"pdfs({q}), pdfs_tmp({q}): {data_type}[3D]", layout=layout)
    macroscopic_fields = {'density': density_field, 'velocity': velocity_field}

    lbm_config = LBMConfig(stencil=stencil, streaming_pattern=streaming_pattern,
                           method=Method.CUMULANT, relaxation_rate=omega,
                           galilean_correction=True, fourth_order_correction=0.1,
                           force_model=ForceModel.GUO, force=force_field.center_vector,
                           subgrid_scale_model=SubgridScaleModel.QR,
                           compressible=True, zero_centered=True,
                           omega_output_field=omega_field,
                           eddy_viscosity_field=eddy_viscosity_field,
                           output=macroscopic_fields)

    lbm_optimisation = LBMOptimisation(field_layout=layout, symbolic_field=pdfs, cse_global=True, cse_pdfs=False)

    if not is_inplace(streaming_pattern):
        lbm_opt = replace(lbm_optimisation, symbolic_temporary_field=pdfs_tmp)
        field_swaps = [(pdfs, pdfs_tmp)]
    else:
        field_swaps = []

    collision_rule = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_optimisation)
    collision_rule = insert_fast_divisions(collision_rule)
    collision_rule = insert_fast_sqrts(collision_rule)

    lb_method = collision_rule.method

    # Welford update for output
    welford_output_update = welford_assignments(field=velocity_field, mean_field=mean_velocity_field,
                                                sum_of_squares_field=sum_of_squares_field,
                                                sum_of_cubes_field=sum_of_cubes_field)
    generate_sweep(ctx, "waLBerlaWind_WelfordOutput", welford_output_update, target=target,
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)

    
    @ps.kernel
    def sos_resetter():
        for d in range(stencil.D**2):
            field_access = sum_of_squares_field.center.at_index(d)
            field_access @= sp.Float(0)
    generate_sweep(ctx, "waLBerlaWind_SoSResetter", ps.AssignmentCollection(sos_resetter), target=target,
    gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)
    
    
    @ps.kernel
    def soc_resetter():
        for d in range(stencil.D**3):
            field_access = sum_of_cubes_field.center.at_index(d)
            field_access @= sp.Float(0)
    generate_sweep(ctx, "waLBerlaWind_SoCResetter", ps.AssignmentCollection(soc_resetter), target=target,
    gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)
    

    welford_nut_update = welford_assignments(field=eddy_viscosity_field, mean_field=mean_eddy_viscosity_field)
    generate_sweep(ctx, "waLBerlaWind_WelfordEddyViscosity", welford_nut_update, target=target,
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)
    welford_strain_update = welford_assignments(field=strain_rate_field, mean_field=mean_strain_rate_field)
    generate_sweep(ctx, "waLBerlaWind_WelfordStrainRate", welford_strain_update, target=target,
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)
    

    #   PDF Setter -> used for initialisation before 0th timestep
    initial_rho = sp.Symbol('rho_0')
    pdfs_setter = pdf_initialization_assignments(lb_method=lb_method,
                                                 density=density_field.center,
                                                 velocity=velocity_field.center_vector,
                                                 pdfs=pdfs,
                                                 streaming_pattern=streaming_pattern, previous_timestep=get_timesteps(streaming_pattern)[0])

    generate_sweep(ctx, "waLBerlaWind_PdfSetter", pdfs_setter, ghost_layers_to_include=ghost_layers)

    #   Macro getter -> used when loading snapshot
    getter_assignments = macroscopic_values_getter(lb_method, velocity=velocity_field.center_vector,
                                                   pdfs=pdfs.center_vector, density=density_field.center)
    generate_sweep(ctx, 'waLBerlaWind_MacroGetter', getter_assignments)

    # MACROSCOPIC FIELD PACK INFOS
    generate_pack_info_for_field(ctx, 'waLBerlaWind_DensityPackInfo', density_field, target=target)
    generate_pack_info_for_field(ctx, 'waLBerlaWind_VelocityPackInfo', velocity_field, target=target)

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

    generate_sweep(ctx, "waLBerlaWind_StrainRateWriter", strain_rate_ac, target=target,
                   gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)
    

    ### FLOW DRIVERS
    generate_flow_driver_collection(ctx, "FlowDriverCollection", force_field=force_field, velocity_field=velocity_field,
                                    nonuniform=False,
                                    target=target, ghost_layers_to_include=ghost_layers, namespace='wind',
                                    gpu_indexing_params=gpu_indexing_params, max_threads=max_threads)

    name = 'waLBerlaWind'
    cpu_vectorise_info = {'nontemporal': True}
    generate_lbm_package(ctx, name=f"{name}_",
                         collision_rule=collision_rule,
                         lbm_config=lbm_config, lbm_optimisation=lbm_optimisation,
                         nonuniform=False, boundaries=[no_slip, free_slip, wfb, uniform_ubb, loglaw_ubb, outflow],
                         macroscopic_fields=macroscopic_fields,
                         target=target, gpu_indexing_params=gpu_indexing_params, max_threads=max_threads,
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

    additional_headers = {
    	"field/Layout.h",
    	"lbm_generated/field/PdfField.h",
    	"lbm_generated/field/AddToStorage.h",
    	"lbm_generated/gpu/GPUPdfField.h",
    	"lbm_generated/gpu/AddToStorage.h",
    	"gpu/AddGPUFieldToStorage.h",
    	"gpu/DeviceSelectMPI.h",
    	"gpu/HostFieldAllocator.h",
    	"gpu/communication/MemcpyPackInfo.h",
    	"gpu/ShiftedPeriodicity.h",
    	"gpu/timing/DeviceSynchronizePolicy.h",
    	"lbm_generated/gpu/UniformGeneratedGPUPdfPackInfo.h",
    	"gpu/communication/UniformGPUScheme.h",
    	"turbine_topology/gpu/Tree.h",
    }

    generate_info_header(ctx, 'waLBerlaWind_KernelInfo',
                         field_typedefs=field_typedefs,
                         additional_headers=additional_headers,
                         additional_code=info_header.format(**info_header_params)
                         )