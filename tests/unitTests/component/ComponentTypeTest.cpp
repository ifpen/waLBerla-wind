
#include "component/ComponentType.h"

#include <core/debug/Debug.h>

#include <cstdlib>

int main() {

    using turbine_core::ComponentType;

    ComponentType none    = ComponentType::NONE;
    ComponentType tower   = ComponentType::TOWER;
    ComponentType hub     = ComponentType::HUB;
    ComponentType nacelle = ComponentType::NACELLE;
    ComponentType blade   = ComponentType::BLADE;
    ComponentType any     = ~ComponentType::NONE;

    WALBERLA_ASSERT(
        !!( none & none ) == false, "`None & None` should be false "
    )

    WALBERLA_ASSERT(
        !!( none & tower   ) ||
        !!( none & hub     ) ||
        !!( nacelle & none ) ||
        !!( blade & none ) == false, "Anything and None should be false."
    )

    WALBERLA_ASSERT(
        !!( tower   & tower   ) &&
        !!( hub     & hub     ) &&
        !!( nacelle & nacelle ) &&
        !!( blade   & blade   ) == true, "Anything and itself should be true."
    )

    WALBERLA_ASSERT(
        !!( tower & hub     ) ||
        !!( tower & nacelle ) ||
        !!( tower & blade   ) == false, ""
    )

    WALBERLA_ASSERT(
        !!( hub & tower   ) ||
        !!( hub & nacelle ) ||
        !!( hub & blade   ) == false, ""
    )

    WALBERLA_ASSERT(
        !!( nacelle & tower   ) ||
        !!( nacelle & hub     ) ||
        !!( nacelle & blade   ) == false, ""
    )

    WALBERLA_ASSERT(
        !!( blade & tower   ) ||
        !!( blade & hub     ) ||
        !!( blade & nacelle ) == false, ""
    )

    WALBERLA_ASSERT(
        !!( tower & hub ) ||
        !!( tower & ( hub | nacelle ) ) ||
        !!( tower & ( hub | nacelle | blade ) ) == false, ""
    )

    WALBERLA_ASSERT(
        !!( tower & ( hub | nacelle | blade | tower ) ) == true, ""
    )

    WALBERLA_ASSERT(
            !!( tower   & any ) &&
            !!( hub     & any ) &&
            !!( nacelle & any ) &&
            !!( blade   & any ) == true, ""
    )

    return EXIT_SUCCESS;

}

