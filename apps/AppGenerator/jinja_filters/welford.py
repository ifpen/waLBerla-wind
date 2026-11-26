from .enums import *


class Welford:

    def __init__(self, basename: str, target: Target, active: bool, grid: Grid,
                 field: Fields, mean_field: Fields, sos_field: Fields = None, soc_field: Fields = None):

        if soc_field:
            assert sos_field, "You cannot specify a Welford with SoC output without SoS output."

        self.basename = basename
        self.target = target
        self.field = field
        self.mean_field = mean_field
        self.sos_field = sos_field
        self.soc_field = soc_field
        self.welford_fields: list[Fields] = [field, mean_field, sos_field, soc_field]
        self.welford_fields = [f for f in self.welford_fields if f is not None]
        self.welford_fields.sort(key=lambda curr_field: str(curr_field))
        self.active = active
        self.grid = grid

    def object_creation(self, runtime_gpu_indexing):
        if not self.active:
            return ""

        result = []

        result.append(f"/// {self.basename.upper()}")
        result.append(f"walberla::pystencils::waLBerlaWind_{self.basename} {self.sweep_name()}(")
        result.append("\t" + "".join([f"{get_field_variable(field, self.target)}, " for field in self.welford_fields]))
        result.append(f"\treal_t(initialWelfordCounter){', gpuBlockSize[0], gpuBlockSize[1], gpuBlockSize[2]' if runtime_gpu_indexing else ''});")

        return "\n".join(result)

    def create_lambda(self):
        sweep_name = self.sweep_name()
        result = []
        result.append(f"auto {self.lambda_name()} = [&{sweep_name}](walberla::IBlock * block) {{")
        result.append(f"    {sweep_name}(block);")
        result.append(f"}};")
        return  "\n".join(result)

    @classmethod
    def read_interval(cls):
        return (f"const uint_t welfordInterval = walberlaEnv.config()->getOneBlock(\"Output\")"
                f".getParameter<uint_t>(\"welfordInterval\", uint_t(0));")

    def increase_counter(self):
        sweep_name = self.sweep_name()
        return f"[&]({'' if self.grid == Grid.UNIFORM else 'uint_t'}) {{ {sweep_name}.setCounter(real_t({sweep_name}.getCounter() + 1)); }}"

    def sweep_name(self):
        variable = self.basename[0].lower() + self.basename[1:]
        return f"{variable}Sweep"

    def lambda_name(self):
        variable = self.basename[0].lower() + self.basename[1:]
        return f"{variable}Lambda"

    def after_timestep_function(self):
        """Counter update/reset followed by the Welford sweep on all blocks, as one void() function.

        Used for refined grids: added with addFuncAfterTimeStep it runs once per coarse time step,
        so all levels are updated with the same counter (as post-collision function it would run
        once per level and substep, which advances the shared counter several times per step)."""
        result = ["[&]() {"]
        result += self._reset_body()
        result.append(f"\tfor ( auto & block : *blocks ) {{")
        result.append(f"\t\t{self.lambda_name()}(&block);")
        result.append(f"\t}}")
        result.append(f"}}")
        return "\n".join(result)

    def reset(self, indent: int):

        indent = " " * indent

        result = [f"[&]({'' if self.grid == Grid.UNIFORM else 'uint_t'}){{"]
        result += self._reset_body()
        result.append(f"}}")

        return f"\n{indent}".join(result)

    def _reset_body(self):

        namespace = "field" if self.target == Target.CPU else "gpu"
        field_type = get_field_type(self.mean_field) if self.target == Target.CPU else "GPUField_T<real_t>"

        result = []

        result.append(f"\tif(welfordInterval && (uint_t({self.sweep_name()}.getCounter()) % welfordInterval == 0)){{")
        result.append(f"\t\t{self.sweep_name()}.setCounter(real_t(1));")
        result.append(f"\t\tfor ( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt ) {{")
        result.append(f"\t\t\tauto dst = blockIt->getData<{field_type}>( {get_field_variable(self.mean_field, self.target)} );")
        result.append(f"\t\t\tconst auto src = blockIt->getData<{field_type}>( {get_field_variable(self.field, self.target)} );")
        result.append(f"\t\t\t{namespace}::fieldCopy( dst, src );")

        basename = self.basename[0].lower()+self.basename[1:]
        if len(self.welford_fields) >= 3:
            result.append(f"\t\t\t{basename}SosResetter(blockIt.get());")
        if len(self.welford_fields) >= 4:
            result.append(f"\t\t\t{basename}SocResetter(blockIt.get());")

        result.append(f"\t\t}}")

        result.append(f"\t}} else {{")
        result.append(f"\t\t{self.sweep_name()}.setCounter(real_t({self.sweep_name()}.getCounter() + 1));")
        result.append(f"\t}}")

        return result
