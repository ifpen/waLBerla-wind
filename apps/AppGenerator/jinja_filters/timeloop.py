from .enums import *

from .snapshot_handler import generate_snapshot_creation
from .welford import Welford

class Timeloop:

    def __init__(self, name: str, grid: Grid, target: Target,
                 field_map: set[Fields], vtk_output: bool, with_snapshot: bool, generate_wfb: bool,
                 turbine_communication: bool, generate_control: bool,
                 welford_output: Welford, welford_eddy_viscosity: Welford, welford_strain_rate: Welford):
        self.name = name
        self.grid = grid
        self.target = target

        self.field_map = field_map
        self.vtk_output = vtk_output
        self.with_snapshot = with_snapshot
        self.generate_wfb = generate_wfb
        self.turbine_communication = turbine_communication
        self.generate_control = generate_control

        self.welford_output = welford_output
        self.welford_eddy_viscosity = welford_eddy_viscosity
        self.welford_strain_rate = welford_strain_rate

        self.recursive_timestep_name = "recursiveTimestep"

    def declaration(self):
        return (f"auto {self.name} = walberla::timeloop::SweepTimeloop<TimingPolicy_T>"
                f"(blocks->getBlockStorage(), timesteps);")

    def single_step(self):
        return f"{self.name}.singleStep(timing);"

    def add_to_timeloop(self):
        if self.grid == Grid.UNIFORM:
            return self._add_to_timeloop_uniform()
        elif self.grid == Grid.STATIC_REFINEMENT or self.grid == Grid.DYNAMIC_REFINEMENT:
            return self._add_to_timeloop_nonuniform()
        else:
            raise ValueError("Unknown grid configuration")

    def add_after_timestep(self, function: tuple[str, str]):
        return f"{self.name}.addFuncAfterTimeStep( {function[1]}, \"{function[0]}\" );"

    def remaining_time_logger(self):
        return f"walberla::timing::RemainingTimeLogger( {self.name}.getNrOfTimeSteps(), remainingTimeLoggerFrequency )"

    def _add_to_timeloop_uniform(self):

        result = []

        result.append(self._add_sweep_uniform(sweep=("Boundary Handling", "boundaryCollection.getSweep()"),
                                              before_fcts=[{"Field communication": "communication->getCommunicateFunctor()"},
                                                           {"Shifted periodicity": "[&boundarySetup, &shiftedPeriodicity]() { if( boundarySetup.inflowType() == InflowSetup::ShiftedPeriodic ) shiftedPeriodicity(); }"}]))

        after_streamcollide = []
        if self.vtk_output:
            after_streamcollide.append({"VTK output": "output"})
        if self.with_snapshot:
            after_streamcollide.append({"Snapshot creation":
                                      generate_snapshot_creation(vtk_output=self.vtk_output,
                                                                 target=self.target,  grid=self.grid,
                                                                 field_map=self.field_map, indent=12,
                                                                 welford_output=self.welford_output,
                                                                 timeloop_name=self.name)})

        result.append(self._add_sweep_uniform(sweep=("LBM stream-collide", "sweepCollection.streamCollide()"), after_fcts=after_streamcollide))

        if self.welford_output.active:
            result.append(self._add_sweep_uniform(sweep=(f"{self.welford_output.basename} Sweep", self.welford_output.lambda_name()),
                                                  before_fcts=[{f"{self.welford_output.basename} Sweep": self.welford_output.reset(19)}]))

        if self.welford_eddy_viscosity.active:
            result.append(self._add_sweep_uniform(sweep=(f"{self.welford_eddy_viscosity.basename} Sweep", self.welford_eddy_viscosity.lambda_name()),
                                                  before_fcts=[{f"{self.welford_eddy_viscosity.basename} Sweep": self.welford_eddy_viscosity.reset(19)}]))

        if self.welford_strain_rate.active:
            result.append(self._add_sweep_uniform(sweep=(f"{self.welford_strain_rate.basename} Sweep", self.welford_strain_rate.lambda_name()),
                                                  before_fcts=[{f"{self.welford_strain_rate.basename} Sweep": self.welford_strain_rate.reset(19)}]))

        after_macro = []
        if self.turbine_communication:
            after_macro.append({"Communicate macroscopic turbine data": "syncMacros"})
        if self.generate_control:
            after_macro.append({"Apply Control": "applyControl"})
        result.append(self._add_sweep_uniform(sweep=("Evaluate density and velocity", "evaluateDensityAndVelocity"),
                                              before_fcts=[{"Rotation": "[&]() { farm.callback(Component::Function::UPDATE_DISCRETISATION); }"}],
                                              after_fcts=after_macro))

        result.append(self._add_sweep_uniform(sweep=("Setting driving force", "flowDriver")))

        before_spread = [{"Calculate forces": "[&]() { farm.callback( Component::Function::CALCULATE_FORCES ); }"}]
        after_spread = []
        if self.turbine_communication:
            before_spread.append({"Communicate force data": "syncForces"})
            after_spread.append({"Reset communication data": "[&]() { farm.callback(Component::Function::RESET_COMMUNICATION_DATA); }"})

        result.append(self._add_sweep_uniform(sweep=("Spread forces", "spreadForces"),
                                              before_fcts=before_spread,
                                              after_fcts=after_spread))

        return "\n".join(result)

    def _add_sweep_uniform(self, sweep: tuple[str, str],
                           before_fcts: list[dict[str, str]] = None,
                           after_fcts: list[dict[str, str]] = None):

        add_str = f"{self.name}.add() "
        indent = " " * len(add_str)

        if before_fcts:
            before_str = []
            for before_dict in before_fcts:
                for before in before_dict.items():
                    before_str.append(f"<< walberla::BeforeFunction({before[1]}, \"{before[0]}\")")
            before_str = f"\n{indent}".join(before_str)
            add_str += before_str

        afters = []
        if after_fcts:
            for after_dict in after_fcts:
                for after in after_dict.items():
                    afters.append(f"<< walberla::AfterFunction({after[1]}, \"{after[0]}\")")
        afters = f"\n{indent}".join(afters)

        results = (add_str + (f"\n{indent}" if before_fcts else '') + f"<< walberla::Sweep({sweep[1]}, \"{sweep[0]}\")"
                   + (f"\n{indent}" if after_fcts else '') + afters)
        return results + ";\n"

    def _add_to_timeloop_nonuniform(self):
        result = []

        # recursive timestep creation
        timestep_type = f"walberla::lbm_generated::BasicRecursiveTimeStep{'GPU' if self.target == Target.GPU else ''}"
        pdf_type = get_field_type(Fields.PDF, self.target) if self.target == Target.GPU else get_field_type(Fields.PDF)
        result.append(f"{timestep_type}<{pdf_type}, SweepCollection_T, BoundaryCollection_T{', walberla::timeloop::SweepTimeloop<TimingPolicy_T>' if self.target == Target.GPU else ''}> {self.recursive_timestep_name} (")
        result.append(f"\tblocks, {get_field_variable(Fields.PDF, self.target)}, sweepCollection, boundaryCollection, communication, pdfPackInfo );\n")

        if self.vtk_output:
            result.append(self._add_function_nonuniform(function=("VTK output", "[&](uint_t level) { if(level!=0) return; turbineOutput.write(); }"), is_sweep=False, level=0))
        if self.with_snapshot:
            result.append(self._add_function_nonuniform(function=("Snapshot creation",
                                                                  generate_snapshot_creation(vtk_output=self.vtk_output,
                                                                                             target=self.target, grid=self.grid,
                                                                                             field_map=self.field_map, indent=12,
                                                                                             welford_output=self.welford_output,
                                                                                             timeloop_name=self.name)), is_sweep=False))


        result.append(self._add_function_nonuniform(function=("Rotation", "[&](uint_t) { farm.callback(Component::Function::UPDATE_DISCRETISATION); }"), is_sweep=False))
        result.append(self._add_function_nonuniform(function=("Evaluate density and velocity", "farm.evaluateDensityAndVelocity"), is_sweep=True))

        if self.turbine_communication:
            result.append(self._add_function_nonuniform(function=("Communicate macroscopic turbine data", " [&](uint_t){farm.syncNextNeighbour(mpi::SYNC_MACRO, 1, false);}"), is_sweep=False))
        if self.generate_control:
            result.append(self._add_function_nonuniform(function=("Apply Control", "[&](uint_t) {farm.applyControl();}"), is_sweep=False))

        result.append(self._add_function_nonuniform(function=("Setting driving force", "flowDriver"), is_sweep=True))
        result.append(self._add_function_nonuniform(function=("Calculate forces", "[&](uint_t) { farm.callback( Component::Function::CALCULATE_FORCES ); }"), is_sweep=False))

        if self.turbine_communication:
            result.append(self._add_function_nonuniform(function=("Communicate force data", "[&](uint_t){farm.syncNextNeighbour(mpi::SYNC_FORCE, maxKernelWidth+uint_t(real_t(diameter)/real_t(2.0)), false);}"), is_sweep=False))

        result.append(self._add_function_nonuniform(function=("Spread forces", "farm.spreadForces"), is_sweep=True))

        if self.turbine_communication:
            result.append(self._add_function_nonuniform(function=("Reset communication data", "[&](uint_t) { farm.callback(Component::Function::RESET_COMMUNICATION_DATA); }"), is_sweep=False))

        result.append(f"\n{self.recursive_timestep_name}.addRefinementToTimeLoop( {self.name} );\n")

        if self.welford_output.active:
            result.append(self.add_after_timestep(function=("Welford Output Sweep", self.welford_output.after_timestep_function())))
        if self.welford_eddy_viscosity.active:
            result.append(self.add_after_timestep(function=("Welford Eddy Viscosity Sweep", self.welford_eddy_viscosity.after_timestep_function())))
        if self.welford_strain_rate.active:
            result.append(self.add_after_timestep(function=("Welford Strain Rate Sweep", self.welford_strain_rate.after_timestep_function())))

        return "\n".join(result)

    def _add_function_nonuniform(self, function: tuple[str, str], is_sweep: bool, level: int=-1):

        if not is_sweep:
            void_function = function[1]
        else:
            level_check = f"if(level != {level}) return;\n" if level != -1 else ""
            void_function = (f"[&](uint_t level) {{\n" + level_check + f"std::vector<walberla::Block* > blocksOnLevel;\nblocks->getBlocks(blocksOnLevel, level);\nfor (auto block: blocksOnLevel) {{\n\t{function[1]}(block);\n}}}}")

        nl = '\n'
        add_string = f'{self.recursive_timestep_name}.addPostCollisionFunction({void_function}, "{function[0]}");{nl if is_sweep else ""}'
        # add_string = f"{self.recursive_timestep_name}.addPostCollisionFunction({void_function}, \"{function[0]}\"{', ' + str(level) if not level == -1 else ''});{nl if is_sweep else ''}"

        return add_string