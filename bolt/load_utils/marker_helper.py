import warp as wp

from bolt.load_utils.converted_objects import SiteData
from bolt.load_utils.osim_types import OSimType
from bolt.load_utils.site_helper import convert_station


def convert_markers(model: OSimType.Model) -> list[SiteData]:
    """ Returns the all the converted sites in the model """
    marker_data = []

    markers = model.getMarkerSet()
    for marker in markers:
        marker = OSimType.Marker.safeDownCast(marker)
        marker_station = convert_station(marker)
        marker_data.append(marker_station)
    return marker_data
