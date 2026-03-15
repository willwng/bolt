import opensim as osim


class OSimType:
    Property = osim.common.AbstractProperty

    Frame = osim.simulation.PhysicalFrame
    PhysicalOffsetFrame = osim.PhysicalOffsetFrame

    Joint = osim.simulation.Joint
    EllipsoidJoint = osim.EllipsoidJoint
    CustomJoint = osim.CustomJoint
    TransformAxis = osim.TransformAxis

    Body = osim.simulation.Body

    Vec3 = osim.simbody.Vec3
    Quat = osim.simbody.Quaternion
    Transform = osim.simbody.Transform
