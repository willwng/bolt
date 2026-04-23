import opensim as osim


class OSimType:
    Model = osim.Model

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
    MultiplierFunction = osim.MultiplierFunction
    MultivariatePolynomialFunction = osim.MultivariatePolynomialFunction

    Body = osim.simulation.Body

    Muscle = osim.simulation.Muscle
    MillardMuscle = osim.Millard2012EquilibriumMuscle
    ThelenMuscle = osim.Thelen2003Muscle
    Path = osim.AbstractGeometryPath
    ScholzPath = osim.Scholz2015GeometryPath
    GeometryPath = osim.GeometryPath
    FunctionBasedPath = osim.FunctionBasedPath
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
    ExponentialContactForce = osim.ExponentialContactForce

    SpringGeneralizedForce = osim.SpringGeneralizedForce
    CoordinateLimitForce = osim.CoordinateLimitForce
    ActivationCoordinateActuator = osim.ActivationCoordinateActuator

    Property = osim.common.AbstractProperty
    PropertyStringList = osim.common.PropertyStringList
