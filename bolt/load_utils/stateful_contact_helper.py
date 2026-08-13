import warp as wp
import xml.etree.ElementTree as ET

from bolt.load_utils.converted_objects import StatefulContactForce, SiteData
from bolt.load_utils.osim_types import OSimType
from bolt.load_utils.physical_frame_helper import wp_transform_from_osim_transform
from bolt.load_utils.xml_helper import extract_bolt_only_objects, extract_float_from_element
from bolt.types_consts import StatefulContact


def _dummy_parse_station(xml_item) -> SiteData:
    station_xml = xml_item.find("Station")
    name = station_xml.attrib["name"]
    socket_parent_frame = station_xml.find("socket_parent_frame").text
    body_name = socket_parent_frame.split("/")[-1]
    location = station_xml.find("location").text
    location = (float(x) for x in location.split())
    return SiteData(
        name=name, body_name=body_name, offset=wp.vec3(*location)
    )


def _exp_parse_station(model: OSimType.Model, exp_contact_force: OSimType.ExponentialContactForce):
    """ Binding is broken, so we just look at the raw xml. fixme: please remove me as soon as the api is fixed """
    model_xml_path = model.getDocumentFileName()
    tree = ET.parse(model_xml_path)
    root = tree.getroot()
    for exp_contact_force_xml in root.iter("ExponentialContactForce"):
        if exp_contact_force_xml.attrib["name"] == exp_contact_force.getName():
            return _dummy_parse_station(exp_contact_force_xml)


def convert_stateful_contacts(
        model: OSimType.Model,
        model_path: str
) -> list[StatefulContactForce]:
    """ Returns the all the converted exponential contact forces in the model """
    exp_contact_force_data = []

    force_set = model.getForceSet()
    exp_contact_forces = filter(lambda f: f.getConcreteClassName() == "ExponentialContactForce", force_set)
    for exp_contact_force in exp_contact_forces:
        exp_contact_force = OSimType.ExponentialContactForce.safeDownCast(exp_contact_force)

        # raw_station = exp_contact_force.getStation()
        station = _exp_parse_station(model, exp_contact_force)

        shape_params = wp.vec3(*exp_contact_force.getExponentialShapeParameters().to_numpy())
        contact_plane_transform = wp_transform_from_osim_transform(exp_contact_force.getContactPlaneTransform())

        exp_contact_force_data.append(
            StatefulContactForce(
                name=exp_contact_force.getName(),
                use_exp_force=True,

                contact_plane_transform=wp_transform_from_osim_transform(exp_contact_force.getContactPlaneTransform()),
                exponential_shape_parameters=shape_params,
                normal_viscosity=exp_contact_force.getNormalViscosity(),
                max_normal_force=exp_contact_force.getMaxNormalForce(),
                friction_elasticity=exp_contact_force.getFrictionElasticity(),
                friction_viscosity=exp_contact_force.getFrictionViscosity(),
                settle_velocity=exp_contact_force.getSettleVelocity(),
                initial_mu_static=exp_contact_force.getInitialMuStatic(),
                initial_mu_kinetic=exp_contact_force.getInitialMuKinetic(),
                station=station,
                margin=0.0,
            )
        )

    # Bolt-only: StatefulHalfspaceContact
    bolt_only_objects = extract_bolt_only_objects(model_path=model_path)
    for obj in bolt_only_objects:
        if obj.tag == "StatefulHalfspaceContact":
            name = obj.get("name")
            station = _dummy_parse_station(obj)
            radius = extract_float_from_element(obj, "radius")

            exp_contact_force_data.append(
                StatefulContactForce(
                    name=name,
                    use_exp_force=False,
                    contact_plane_transform=wp.transform(
                        wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2.0)),
                    exponential_shape_parameters=wp.vec3(radius, radius, radius),
                    normal_viscosity=extract_float_from_element(obj, "normal_viscosity"),
                    max_normal_force=extract_float_from_element(obj, "stiffness"),
                    friction_elasticity=extract_float_from_element(obj, "friction_elasticity"),
                    friction_viscosity=extract_float_from_element(obj, "friction_viscosity"),
                    settle_velocity=extract_float_from_element(obj, "settle_velocity"),
                    initial_mu_static=extract_float_from_element(obj, "initial_mu_static"),
                    initial_mu_kinetic=extract_float_from_element(obj, "initial_mu_kinetic"),
                    station=station,
                    margin=radius,
                )
            )

    return exp_contact_force_data


def flatten_sites(stl_contact_forces: list[StatefulContactForce]) -> list[SiteData]:
    """ Returns a flattened list of all the sites used by the stateful contact forces. """
    return [contact_force.station for contact_force in stl_contact_forces]


def create_stateful_contact_data(
        stateful_contact_data: list[StatefulContactForce],
        site_start_stl: int,
        body_ordering: dict[str, int]
) -> list[StatefulContact]:
    stl_contacts = []
    for i, data in enumerate(stateful_contact_data):
        contact = StatefulContact()

        contact.contact_plane_transform = data.contact_plane_transform
        contact.shape_parameters = data.exponential_shape_parameters
        contact.normal_viscosity = data.normal_viscosity
        contact.max_normal_force = data.max_normal_force
        contact.friction_elasticity = data.friction_elasticity
        contact.friction_viscosity = data.friction_viscosity
        contact.settle_velocity = data.settle_velocity
        contact.initial_mu_static = data.initial_mu_static
        contact.initial_mu_kinetic = data.initial_mu_kinetic
        contact.margin = data.margin
        contact.use_exp_force = data.use_exp_force

        contact.siteid = site_start_stl + i
        contact.bodyid = body_ordering[data.station.body_name]
        contact.station_B = data.station.offset

        stl_contacts.append(contact)
    return stl_contacts
