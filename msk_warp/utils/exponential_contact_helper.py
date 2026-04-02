import warp as wp

from msk_warp import ExponentialContact
from msk_warp.utils.converted_objects import ExponentialContactForce, SiteData
from msk_warp.utils.osim_types import OSimType
from msk_warp.utils.physical_frame_helper import wp_transform_from_osim_transform


def _dummy_parse_station(model: OSimType.Model, exp_contact_force: OSimType.ExponentialContactForce):
    """ Binding is broken, so we just look at the raw xml. fixme: please remove me as soon as the api is fixed """
    import xml.etree.ElementTree as ET
    model_xml_path = model.getDocumentFileName()
    tree = ET.parse(model_xml_path)
    root = tree.getroot()
    for exp_contact_force_xml in root.iter("ExponentialContactForce"):
        if exp_contact_force_xml.attrib["name"] == exp_contact_force.getName():
            station_xml = exp_contact_force_xml.find("Station")
            name = station_xml.attrib["name"]
            socket_parent_frame = station_xml.find("socket_parent_frame").text
            body_name = socket_parent_frame.split("/")[-1]
            location = station_xml.find("location").text
            location = (float(x) for x in location.split())
            return SiteData(
                name=name, body_name=body_name, offset=wp.vec3(*location)
            )


def convert_exponential_contacts(model: OSimType.Model) -> list[ExponentialContactForce]:
    """ Returns the all the converted exponential contact forces in the model """
    exp_contact_force_data = []

    force_set = model.getForceSet()
    exp_contact_forces = filter(lambda f: f.getConcreteClassName() == "ExponentialContactForce", force_set)
    for exp_contact_force in exp_contact_forces:
        exp_contact_force = OSimType.ExponentialContactForce.safeDownCast(exp_contact_force)

        # raw_station = exp_contact_force.getStation()
        station = _dummy_parse_station(model, exp_contact_force)

        exp_contact_force_data.append(
            ExponentialContactForce(
                name=exp_contact_force.getName(),
                contact_plane_transform=wp_transform_from_osim_transform(exp_contact_force.getContactPlaneTransform()),
                exponential_shape_parameters=wp.vec3(*exp_contact_force.getExponentialShapeParameters().to_numpy()),
                normal_viscosity=exp_contact_force.getNormalViscosity(),
                max_normal_force=exp_contact_force.getMaxNormalForce(),
                friction_elasticity=exp_contact_force.getFrictionElasticity(),
                friction_viscosity=exp_contact_force.getFrictionViscosity(),
                settle_velocity=exp_contact_force.getSettleVelocity(),
                initial_mu_static=exp_contact_force.getInitialMuStatic(),
                initial_mu_kinetic=exp_contact_force.getInitialMuKinetic(),
                station=station,
            )
        )

    return exp_contact_force_data


def flatten_sites(exp_contact_forces: list[ExponentialContactForce]) -> list[SiteData]:
    """ Returns a flattened list of all the sites used by the exponential contact forces. """
    return [exp_contact_force.station for exp_contact_force in exp_contact_forces]


def create_exp_contact_data(
        exp_contact_data: list[ExponentialContactForce],
        site_start_exp: int,
        body_ordering: dict[str, int]
) -> list[ExponentialContact]:
    exp_contacts = []
    for i, data in enumerate(exp_contact_data):
        contact = ExponentialContact()

        contact.contact_plane_transform = data.contact_plane_transform
        contact.shape_parameters = data.exponential_shape_parameters
        contact.normal_viscosity = data.normal_viscosity
        contact.max_normal_force = data.max_normal_force
        contact.friction_elasticity = data.friction_elasticity
        contact.friction_viscosity = data.friction_viscosity
        contact.settle_velocity = data.settle_velocity
        contact.initial_mu_static = data.initial_mu_static
        contact.initial_mu_kinetic = data.initial_mu_kinetic

        contact.siteid = site_start_exp + i
        contact.bodyid = body_ordering[data.station.body_name]
        contact.station_B = data.station.offset

        exp_contacts.append(contact)
    return exp_contacts
