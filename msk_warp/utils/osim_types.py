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

    Muscle = osim.simulation.Muscle
    MillardMuscle = osim.Millard2012EquilibriumMuscle
    ScholzPath = osim.Scholz2015GeometryPath
    PathPoint = osim.PathPoint

    Vector = osim.Vector
    Vec3 = osim.simbody.Vec3
    Quat = osim.simbody.Quaternion
    Transform = osim.simbody.Transform

    Mesh = osim.Mesh
    Station = osim.Station

    ContactGeometry = osim.ContactGeometry
    ContactSphere = osim.ContactSphere
    ContactEllipsoid = osim.ContactEllipsoid
    ContactHalfSpace = osim.ContactHalfSpace

    CoordinateLinearDamper = osim.CoordinateLinearDamper
    CoordinateLinearStop = osim.CoordinateLinearStop
    CoordinateLinearSpring = osim.CoordinateLinearSpring
    SpringGeneralizedForce = osim.SpringGeneralizedForce
    CoordinateLimitForce = osim.CoordinateLimitForce

    Property = osim.common.AbstractProperty
    PropertyStringList = osim.common.PropertyStringList
