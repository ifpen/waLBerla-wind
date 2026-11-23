
#include <cstdlib>

#include <core/Environment.h>
#include <core/debug/TestSubsystem.h>
#include <core/DataTypes.h>

#include <field/Field.h>
#include <field/GhostLayerField.h>
#include <field/Layout.h>

#include "walberla_helper/field/Field.h"

namespace field_test {

    using namespace walberla;

    const field::Layout layout{ field::Layout::fzyx };

    const cell_idx_t xs = 2;
    const cell_idx_t ys = 4;
    const cell_idx_t zs = 8;
    const cell_idx_t fs = 3;

    void compareAgainstFieldTest() {

        Field<int, fs> standardField(xs, ys, zs, layout);

        // initialise field with random data
        for( auto it = standardField.begin(); it != standardField.end(); ++it ) {
            *it = std::rand();
        }

        // extract field info to custom field
        turbine_core::field::Field<int> customField(&standardField);

        // check for equality of data
        for( cell_idx_t f = 0; f < fs; ++f ) {
            for( cell_idx_t z = 0; z < zs; ++z ) {
                for( cell_idx_t y = 0; y < ys; ++y ) {
                    for( cell_idx_t x = 0; x < xs; ++x ) {
                        WALBERLA_CHECK_EQUAL( standardField.get(x,y,z,f), customField.get(x,y,z,f) );
                    }
                }
            }
        }
    }

    void compareAgainstGhostFieldTest( const uint_t nGhostLayers ) {

        GhostLayerField<int, fs> standardField(xs, ys, zs, nGhostLayers, 0, layout);

        // initialise field with random data
        for( auto it = standardField.begin(); it != standardField.end(); ++it ) {
            *it = std::rand();
        }

        // extract field info to custom field
        turbine_core::field::Field<int> customField(&standardField);

        // check for equality of data
        for( cell_idx_t f = 0; f < fs; ++f ) {
            for( cell_idx_t z = 0; z < zs; ++z ) {
                for( cell_idx_t y = 0; y < ys; ++y ) {
                    for( cell_idx_t x = 0; x < xs; ++x ) {
                        WALBERLA_CHECK_EQUAL( standardField.get(x,y,z,f), customField.get(x,y,z,f) );
                    }
                }
            }
        }

    }

    //TODO setup test case for GPUField

    int main( int argc, char** argv ) {
        Environment walberlaEnv( argc, argv );

        debug::enterTestMode();

        compareAgainstFieldTest();

        compareAgainstGhostFieldTest(0);
        compareAgainstGhostFieldTest(1);
        compareAgainstGhostFieldTest(2);
        compareAgainstGhostFieldTest(4);

        return EXIT_SUCCESS;
    }

}

int main( int argc, char** argv ) {
    return field_test::main(argc, argv);
}