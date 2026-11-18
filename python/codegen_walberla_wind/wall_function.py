from lbmpy_walberla.additional_data_handler import WFBAdditionalDataHandler


# Workaround for the WFB link selection in lbmpy_walberla
class WallLinkWFBAdditionalDataHandler(WFBAdditionalDataHandler):

    @property
    def stencil_info(self):
        stencil_info = []
        for i, d in enumerate(self._stencil):
            if any(n != 0 and c == -n for n, c in zip(self._normal_direction, d)):
                direction = d if self._dim == 3 else d + (0,)
                stencil_info.append((i, direction, ", ".join(str(e) for e in direction)))
        return stencil_info
