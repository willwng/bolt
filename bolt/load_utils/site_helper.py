import warp as wp

from bolt.load_utils.converted_objects import SiteData
from bolt.load_utils.osim_types import OSimType
from bolt.load_utils.property_helper import extract_vec3
from bolt.load_utils.physical_frame_helper import extract_frame_transform_from_base_frame, get_body_name_of_frame


def convert_station(station: OSimType.Station) -> SiteData:
    station_name = station.getName()
    parent_frame = station.getParentFrame()
    body_name = get_body_name_of_frame(parent_frame)

    # Compute the offset from the body frame
    frame_transform = extract_frame_transform_from_base_frame(parent_frame)
    location = wp.vec3(extract_vec3(station.get_location()))
    offset = wp.transform_point(frame_transform, location)

    return SiteData(
        name=station_name, body_name=body_name, offset=offset
    )


def convert_sites(model: OSimType.Model) -> list[SiteData]:
    """ Returns the all the converted sites in the model """
    site_data = []

    body_list = model.getBodyList()
    for body in body_list:
        components = body.getComponentsList()
        station_components = filter(lambda c: isinstance(c, OSimType.Station), components)
        for station in station_components:
            site_data.append(convert_station(station))
    return site_data


def get_site_body_name(site_data_list: list[SiteData]) -> list[str]:
    return [site_data.body_name for site_data in site_data_list]


def get_site_offset(site_data_list: list[SiteData]) -> list[wp.vec3]:
    return [site_data.offset for site_data in site_data_list]
