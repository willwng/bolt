import warp as wp
from msk_warp.utils.converted_objects import FunctionData, LinearFunctionData, PolynomialFunctionData, \
    ConstantFunctionData, TransformAxisData
from msk_warp.utils.property_helper import extract_vector
from msk_warp.utils.osim_types import OSimType
from msk_warp.utils.python_util import exclusive_sum
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
    poly_coeffs_adr = exclusive_sum(poly_coeffs_num)
    return poly_coeffs_num, poly_coeffs_adr
