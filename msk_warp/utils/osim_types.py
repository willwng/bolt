import opensim as osim


class OSimType:
    Frame = osim.simulation.PhysicalFrame
    PhysicalOffsetFrame = osim.PhysicalOffsetFrame

    Joint = osim.simulation.Joint
    EllipsoidJoint = osim.EllipsoidJoint
    CustomJoint = osim.CustomJoint

    TransformAxis = osim.TransformAxis
    Function = osim.Function
    LinearFunction = osim.LinearFunction
    ConstantFunction = osim.Constant
    PolynomialFunction = osim.PolynomialFunction

    Body = osim.simulation.Body

    Vector = osim.Vector
    Vec3 = osim.simbody.Vec3
    Quat = osim.simbody.Quaternion
    Transform = osim.simbody.Transform

    Mesh = osim.Mesh

    Property = osim.common.AbstractProperty
    PropertyStringList = osim.common.PropertyStringList
