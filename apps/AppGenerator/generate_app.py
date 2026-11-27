import os
import shutil
import argparse
import json

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lbmpy import Stencil
from pystencils_walberla import CodeGeneration

from jinja_filters.enums import *
from jinja_filters.jinja_filters import add_turbine_filters_to_jinja_env

#######################################################################################################################
### START USER-DEFINED CONFIGURATION ##################################################################################
#######################################################################################################################

stencil: Stencil = Stencil.D3Q27
turbine_communication: bool = True
turbine_output: bool = True
vtk_output: bool = True
field_map: set[Fields] = {Fields.MEAN_VELOCITY,
                          Fields.SUM_OF_SQUARES,
                          Fields.SUM_OF_CUBES,
                          Fields.STRAIN_RATE,
                          Fields.MEAN_STRAIN_RATE,
                          Fields.EDDY_VISCOSITY,
                          Fields.MEAN_EDDY_VISCOSITY}
with_snapshot: bool = True
generate_wfb: bool = True
generate_control: bool = True
streaming_pattern: StreamingPattern = StreamingPattern.PULL
runtime_gpu_indexing: bool = False
add_likwid_markers: bool = False

#######################################################################################################################
### END USER-DEFINED CONFIGURATION ####################################################################################
#######################################################################################################################

# sanity checks of user input
assert stencil in [Stencil.D3Q19, Stencil.D3Q27], f"waLBerla-wind is currently not supported for {stencil} stencils."


# add defaults

if not field_map:
    field_map = set()

field_map.update([Fields.PDF, Fields.DENSITY, Fields.VELOCITY, Fields.FORCE, Fields.FLAG])

if Fields.SUM_OF_CUBES in field_map:
    field_map.add(Fields.SUM_OF_SQUARES)

if Fields.SUM_OF_SQUARES in field_map:
    field_map.add(Fields.MEAN_VELOCITY)

if Fields.EDDY_VISCOSITY in field_map or Fields.STRAIN_RATE in field_map:
    field_map.add(Fields.OMEGA)

if Fields.MEAN_EDDY_VISCOSITY in field_map:
    field_map.add(Fields.EDDY_VISCOSITY)

if Fields.MEAN_STRAIN_RATE in field_map:
    field_map.add(Fields.STRAIN_RATE)

########################################################################################################################

PARSE_HELPER = {"on":  True,  "1": True,  "yes": True,  "true":  True,
                "off": False, "0": False, "no":  False, "false": False}

class TurbineCodeGeneration(CodeGeneration):
    def __init__(self):
        super(CodeGeneration)
        parser = argparse.ArgumentParser(description='Code Generation script for waLBerla.')
        parser.add_argument('-f', '--files', nargs='*',
                            help='List all files that will be generated with absolute path',
                            default=[])
        parser.add_argument('-c', '--cmake-args', type=json.loads,
                            help='Provide CMake configuration (will be used in the codegen config)')
        parser.add_argument('-l', '--list-only',
                            help="Script will not generate files but list files it would generated without this option")
        args = parser.parse_args()

        cmake_args = {key: PARSE_HELPER.get(str(value).lower(), value) for key, value in args.cmake_args.items()}

        self.context = TurbineGenerationContext(cmake_args)
        self.expected_files = args.files
        self.list_only = True if args.list_only else False


class TurbineGenerationContext:
    def __init__(self, cmake_vars):
        self.files_written = []
        self.openmp = cmake_vars['WALBERLA_BUILD_WITH_OPENMP']
        self.optimize_for_localhost = cmake_vars['WALBERLA_OPTIMIZE_FOR_LOCALHOST']
        self.mpi = cmake_vars['WALBERLA_BUILD_WITH_MPI']
        self.double_accuracy = cmake_vars['WALBERLA_DOUBLE_ACCURACY']
        self.cuda = cmake_vars['WALBERLA_BUILD_WITH_CUDA']
        self.hip = cmake_vars['WALBERLA_BUILD_WITH_HIP']
        self.gpu = self.cuda or self.hip
        self.architecture = cmake_vars['CODEGEN_ARCHITECTURE']
        self.grid = cmake_vars['CODEGEN_GRID']
        self.source_dir = cmake_vars['CMAKE_CURRENT_SOURCE_DIR']

    def write_file(self, name, content):
        self.files_written.append(os.path.abspath(name))
        with open(name, 'w') as f:
            f.write(content)

########################################################################################################################

with TurbineCodeGeneration() as ctx:
    target = Target.GPU if ctx.architecture == "gpu" else Target.CPU
    grid = Grid.to_enum(ctx.grid)
    cwd = ctx.source_dir

    if target == Target.CPU:
        runtime_gpu_indexing = False

    if target == Target.GPU and Grid.is_refined(grid) and StreamingPattern.is_inplace(streaming_pattern):
        print('\033[91m' + "Due to a bug, in-place streaming patterns are currently not supported for mesh refinement on GPU. "
              "Please chose a different streaming pattern. We are working on this issue." + '\033[0m')
        exit()

    # copy turbine data into destination repository
    creation_path = os.path.abspath(".")  # os.path.abspath(f"GeneratedApp_{ctx.architecture}_{ctx.grid}") # os.path.normpath(os.path.join(cwd, "..", f"GeneratedApp_{config_tokens[0]}_{config_tokens[1]}"))
    data_path = os.path.join(cwd, "templates", "turbine_example")
    turbine_data = os.listdir(data_path)
    for data in turbine_data:
        full_path = os.path.join(data_path, data)
        if os.path.isdir(full_path):
            shutil.copytree(full_path, os.path.join(creation_path, data), dirs_exist_ok=True)
        if os.path.isfile(full_path):
            shutil.copy(full_path, creation_path)

    # create application files
    template_loader = FileSystemLoader(searchpath=os.path.join(cwd, "templates"))
    env = Environment(loader=template_loader, undefined=StrictUndefined)
    add_turbine_filters_to_jinja_env(env, env.globals)

    source_extension = source_extension(target)

    jinja_context = {
        'stencil': stencil,
        'target': target,
        'grid': grid,
        'source_extension': source_extension,
        'turbine_communication': turbine_communication,
        'turbine_output': turbine_output,
        'vtk_output': vtk_output,
        'with_snapshot': with_snapshot,
        'generate_wfb': generate_wfb,
        'generate_control': generate_control,
        'streaming_pattern': streaming_pattern,
        'field_map': field_map,
        'welford_output_name': "WelfordOutput",
        'welford_eddy_viscosity_name': "WelfordEddyViscosity",
        'welford_strain_rate_name': "WelfordStrainRate",
        'runtime_gpu_indexing': runtime_gpu_indexing,
        'add_likwid_markers': add_likwid_markers
    }

    cmake_lists = env.get_template("CMakeLists.tmpl.txt").render(**jinja_context)
    main = env.get_template("main.tmpl.cpp").render(**jinja_context)
    generation_script = env.get_template("GenerationScript.tmpl.py").render(**jinja_context)
    input_file = env.get_template("input.tmpl.prm").render(**jinja_context)

    ctx.write_file("CMakeLists.txt", cmake_lists)
    ctx.write_file(f"main.{source_extension}", main)
    ctx.write_file(f"LBMGeneration.py", generation_script)
    ctx.write_file(f"input.prm", input_file)

    source_directory = os.path.join(cwd, "..", os.path.basename(os.path.normpath(creation_path)))
    shutil.copytree(creation_path, source_directory, dirs_exist_ok=True)
