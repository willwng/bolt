import opensim as osim
import itertools

from math import comb
from msk_warp import MAX_POLY_NUM_DOFS, POLY_TILE_SIZE, PolyInts
from msk_warp.utils.osim_types import OSimType
from msk_warp.utils.converted_objects import MuscleFunctionPathData
from msk_warp.utils.muscle_helper import get_muscles
from msk_warp.utils.python_util import remove_slash_prefix, pad_list, exclusive_scan
from msk_warp.utils.property_helper import extract_vector

PADDED_DOF = "__PADDED_DOF"


def polynomial_exponents(dimension: int, order: int):
    """ Generates all combinations of exponents for a multivariate polynomial of given order and number of variables."""
    for exps in itertools.product(range(order + 1), repeat=dimension):
        if sum(exps) <= order:
            yield exps


def pad_exponents_for_max_dimension(exponents: list[tuple], pad_value: int) -> list[PolyInts]:
    """ Pads a list of exponent so that each exponent has a value for each variable up to MAX_POLY_NUM_DOFS. """
    padded_exps = []
    for exp in exponents:
        padded_exp = pad_list(lst=list(exp), target_length=MAX_POLY_NUM_DOFS, pad_value=pad_value)
        padded_exps.append(PolyInts(*padded_exp))
    return padded_exps


def parse_function_based_paths(model_path: str, function_based_path_file: str) -> list[MuscleFunctionPathData]:
    """ Converts the muscle paths in the given model to function-based paths using the provided file """
    muscle_function_path_data = []

    # Load the model and process it with the FunctionBasedPath processor
    processor = osim.ModelProcessor(model_path)
    processor.append(osim.ModOpReplacePathsWithFunctionBasedPaths(function_based_path_file))
    model = processor.process()

    # Now fetch the muscle paths
    muscles = get_muscles(model)
    for muscle in muscles:
        muscle = OSimType.Muscle.safeDownCast(muscle)
        muscle_path = OSimType.FunctionBasedPath.safeDownCast(muscle.getPath())
        if muscle_path is None:
            raise ValueError(f"Muscle {muscle.getName()} does not have a FunctionBasedPath after processing")

        muscle_path_name = f"{muscle.getName()}_{muscle_path.getName()}"

        # Dependent coordinates, pad to MAX_POLY_NUM_DOFS
        coordinates = muscle_path.getCoordinatePaths()
        coordinates = [remove_slash_prefix(coord) for coord in coordinates]
        coordinates = pad_list(coordinates, target_length=MAX_POLY_NUM_DOFS, pad_value=PADDED_DOF)

        # Coefficients
        length_function = muscle_path.get_length_function()
        length_function = OSimType.MultivariatePolynomialFunction.safeDownCast(length_function)
        coefficients = extract_vector(length_function.get_coefficients())
        dimension = length_function.getDimension()
        order = length_function.getOrder()

        # Build exponents, pad to the tile size
        exponents = list(polynomial_exponents(dimension, order))
        exponents = pad_exponents_for_max_dimension(exponents, 0)

        # some checks
        num_expected_terms = comb(dimension + order, order)
        if len(coefficients) != num_expected_terms:
            raise ValueError(f"Num coefficients {len(coefficients)} does not match expected {num_expected_terms}")
        if len(exponents) != num_expected_terms:
            raise ValueError(f"Num exponents {len(exponents)} does not match expected {num_expected_terms}")
        if dimension > MAX_POLY_NUM_DOFS:
            raise ValueError(f"Polynomial dimension {dimension} is greater than max supported {MAX_POLY_NUM_DOFS}")

        # Now we need to pad everything so it can be processed using tiles, fill up to next multiple with dummy data
        n_terms = len(coefficients)
        n_padded_terms = ((n_terms + POLY_TILE_SIZE - 1) // POLY_TILE_SIZE) * POLY_TILE_SIZE
        coefficients = pad_list(coefficients, target_length=n_padded_terms, pad_value=0.0)
        exponents = pad_list(exponents, target_length=n_padded_terms, pad_value=PolyInts(0))

        muscle_function_path_data.append(
            MuscleFunctionPathData(
                name=muscle_path_name,
                coordinates=coordinates,
                coefficients=coefficients,
                exponents=exponents
            )
        )
    return muscle_function_path_data


def get_fn_path_term_coeffs(muscle_function_paths: list[MuscleFunctionPathData]) -> list[float]:
    """ Gets the coefficients of the polynomial terms for all the muscle function paths """
    coeffs = []
    for muscle_path in muscle_function_paths:
        coeffs.extend(muscle_path.coefficients)
    return coeffs


def get_fn_path_term_exps(muscle_function_paths: list[MuscleFunctionPathData]) -> list[PolyInts]:
    """ Gets the exponents of the polynomial terms for all the muscle function paths """
    exps = []
    for muscle_path in muscle_function_paths:
        exps.extend(muscle_path.exponents)
    return exps


def compute_fn_path_term_start_and_count(
        muscle_function_paths: list[MuscleFunctionPathData]
) -> tuple[
    list[int], list[int]]:
    """ Computes the start index and count of polynomial terms for each muscle function path """
    term_count = [len(muscle_path.coefficients) for muscle_path in muscle_function_paths]
    term_start = exclusive_scan(term_count)
    return term_start, term_count


def get_fn_term_adr(
        muscle_function_paths: list[MuscleFunctionPathData],
        ordering: dict[str, int]
) -> list[PolyInts]:
    """ Gets the addresses of the coordinates for each muscle function path """
    term_fn_adr = []

    def _convert_coord_to_adr(c: str) -> int:
        if c == PADDED_DOF:
            return 0
        elif c in ordering:
            return ordering[c]
        else:
            raise ValueError(f"Coordinate {c} in muscle function path not found in ordering")

    for muscle_path in muscle_function_paths:
        coord_adr = [_convert_coord_to_adr(c) for c in muscle_path.coordinates]
        term_fn_adr.append(PolyInts(*coord_adr))

    return term_fn_adr
