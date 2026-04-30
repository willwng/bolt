import warp as wp
from msk_warp.utils.converted_objects import FunctionData, LinearFunctionData, PolynomialFunctionData, \
    ConstantFunctionData, SimmSplineData, TransformAxisData
from msk_warp.utils.property_helper import extract_vector
from msk_warp.utils.osim_types import OSimType
from msk_warp.utils.python_util import exclusive_scan
from typing import Type, TypeVar

T = TypeVar("T")


def convert_function(function: OSimType.Function) -> FunctionData:
    function_class = function.getConcreteClassName()
    if function_class == "LinearFunction":
        linear_function = OSimType.LinearFunction.safeDownCast(function)
        slope = linear_function.getSlope()
        intercept = linear_function.getIntercept()
        return LinearFunctionData(slope=slope, intercept=intercept)
    elif function_class == "Constant":
        constant_function = OSimType.ConstantFunction.safeDownCast(function)
        value = constant_function.getValue()
        return ConstantFunctionData(value=value)
    elif function_class == "PolynomialFunction":
        polynomial_function = OSimType.PolynomialFunction.safeDownCast(function)
        coefficients = extract_vector(polynomial_function.getCoefficients())
        return PolynomialFunctionData(coefficients=coefficients)
    elif function_class == "SimmSpline":
        simm_spline_function = OSimType.SimmSpline.safeDownCast(function)
        x = extract_vector(simm_spline_function.getX().getAsVector())
        y = extract_vector(simm_spline_function.getY().getAsVector())
        return SimmSplineData(x=x, y=y)
    elif function_class == "MultiplierFunction":
        multiplier_function = OSimType.MultiplierFunction.safeDownCast(function)
        scale = multiplier_function.getScale()
        inner_function_data = convert_function(multiplier_function.getFunction())
        scaled_inner_function_data = inner_function_data.scale(scale)
        return scaled_inner_function_data
    else:
        raise ValueError(f"Unsupported function type: {function_class}")


def get_functions_of_type(
        transform_axes: list[TransformAxisData],
        cls: Type[T],
) -> tuple[list[T], list[int]]:
    """ Returns all functions of class [cls], and a list containing the index in the original list """
    ret_fns = [f.function for f in transform_axes if isinstance(f.function, cls)]
    ret_ids = [i for i, f in enumerate(transform_axes) if isinstance(f.function, cls)]
    return ret_fns, ret_ids


def get_linear_fn_mb(linear_fns: list[LinearFunctionData]) -> list[wp.vec2]:
    """ Returns a contiguous list of wp.vec2 containing the slopes and intercepts of linear functions """
    return [wp.vec2(fn.slope, fn.intercept) for fn in linear_fns]


def get_const_fn_vals(const_fns: list[ConstantFunctionData]) -> list[float]:
    """ Returns a contiguous list of the values for constant functions """
    return [fn.value for fn in const_fns]


def get_flattened_poly_coeffs(poly_fns: list[PolynomialFunctionData]) -> list[float]:
    """ Returns a flattened list of the coefficients for polynomial functions. The coefficients are flattened in order, meaning all coefficients for the first function come first, followed by all coefficients for the second function, and so on. """
    flattened_coeffs = []
    for fn in poly_fns:
        flattened_coeffs.extend(fn.coefficients)
    return flattened_coeffs


def get_poly_coeffs_num_adr(poly_fns: list[PolynomialFunctionData]) -> tuple[list[int], list[int]]:
    """ Returns the number of the coefficients and the starting address for each polynomial function """
    poly_coeffs_num = [len(fn.coefficients) for fn in poly_fns]
    poly_coeffs_adr = exclusive_scan(poly_coeffs_num)
    return poly_coeffs_num, poly_coeffs_adr


def get_spline_xy_y2s(spline_fns: list) -> list[wp.vec3]:
    """
    Returns a flattened list of the (x, y, y2) triplets for simm spline functions,
    where y2 is the precomputed second derivative for the Natural Cubic Spline.
    """
    flattened_xy_y2s = []

    for fn in spline_fns:
        n = len(fn.x)
        y2 = [0.0] * n
        if n > 2:
            # Thomas algorithm for solving the tridiagonal system
            c_prime = [0.0] * n
            d_prime = [0.0] * n
            # Forward elimination
            for i in range(1, n - 1):
                hx_prev = fn.x[i] - fn.x[i - 1]
                hx_next = fn.x[i + 1] - fn.x[i]
                a = hx_prev
                b = 2.0 * (hx_prev + hx_next)
                c = hx_next
                dy_prev = (fn.y[i] - fn.y[i - 1]) / hx_prev
                dy_next = (fn.y[i + 1] - fn.y[i]) / hx_next
                d = 6.0 * (dy_next - dy_prev)

                denom = b - a * c_prime[i - 1]
                # prevent division by zero in case of degenerate identical x values
                if denom == 0.0:
                    denom = 1e-7
                c_prime[i] = c / denom
                d_prime[i] = (d - a * d_prime[i - 1]) / denom

            # back substitution
            # natural spline boundary condition: y2[n-1] is inherently 0.0
            for i in range(n - 2, 0, -1):
                y2[i] = d_prime[i] - c_prime[i] * y2[i + 1]

        for x, y, y_sec in zip(fn.x, fn.y, y2):
            flattened_xy_y2s.append(wp.vec3(x, y, y_sec))

    return flattened_xy_y2s


def get_spline_xys_num_adr(spline_fns: list[SimmSplineData]) -> tuple[list[int], list[int]]:
    """ Returns the number of (x, y) pairs and the starting address for each simm spline function """
    spline_xy_num = [len(fn.x) for fn in spline_fns]
    spline_xy_adr = exclusive_scan(spline_xy_num)
    return spline_xy_num, spline_xy_adr
