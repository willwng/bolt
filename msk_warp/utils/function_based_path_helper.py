import opensim as osim
import itertools

from math import comb
from msk_warp import MAX_POLY_NUM_DOFS, MAX_POLY_ORDER, POLY_TILE_SIZE, SUPPORTED_DIM_ORDER, PolyInts
from msk_warp.utils.osim_types import OSimType
from msk_warp.utils.converted_objects import MuscleFunctionPathData, USE_POINT_PATH, PADDED_DOF
from msk_warp.utils.muscle_helper import get_muscles
from msk_warp.utils.python_util import remove_slash_prefix, pad_list, exclusive_scan
from msk_warp.utils.property_helper import extract_vector


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


def parse_function_based_paths(
        model_path: str,
        function_based_path_file: str
) -> list[MuscleFunctionPathData]:
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
        # We weren't given a function based path for this muscle, resort to point path
        if muscle_path is None:
            muscle_function_path_data.append(USE_POINT_PATH)
            continue

        muscle_name = muscle.getName()
        muscle_path_name = f"{muscle_name}_{muscle_path.getName()}"

        # Dependent coordinates
        coordinates = muscle_path.getCoordinatePaths()
        coordinates = [remove_slash_prefix(coord) for coord in coordinates]
        # Coefficients
        length_function = muscle_path.get_length_function()
        length_function = OSimType.MultivariatePolynomialFunction.safeDownCast(length_function)
        coefficients = extract_vector(length_function.get_coefficients())
        dimension = length_function.getDimension()
        order = length_function.getOrder()
        # Exponents
        exponents = list(polynomial_exponents(dimension, order))

        # some checks
        num_expected_terms = comb(dimension + order, order)
        if len(coefficients) != num_expected_terms:
            raise ValueError(f"Num coefficients {len(coefficients)} does not match expected {num_expected_terms}")
        if len(exponents) != num_expected_terms:
            raise ValueError(f"Num exponents {len(exponents)} does not match expected {num_expected_terms}")
        if dimension > MAX_POLY_NUM_DOFS:
            raise ValueError(f"Polynomial dimension {dimension} is greater than max supported {MAX_POLY_NUM_DOFS}")
        if order > MAX_POLY_ORDER:
            raise ValueError(f"Polynomial order {order} is greater than max supported {MAX_POLY_ORDER}")
        if (dimension, order) not in SUPPORTED_DIM_ORDER:
            raise ValueError(f"dimension {dimension} and order {order} are not supported. Please generate new funcs")


        # Pad everything to the max dimension and order
        exponents = pad_exponents_for_max_dimension(exponents, 0)
        coordinates = pad_list(coordinates, target_length=MAX_POLY_NUM_DOFS, pad_value=PADDED_DOF)

        # Now we need to pad everything so it can be processed using tiles
        n_terms = len(coefficients)
        n_padded_terms = ((n_terms + POLY_TILE_SIZE - 1) // POLY_TILE_SIZE) * POLY_TILE_SIZE
        n_tiles = n_padded_terms // POLY_TILE_SIZE
        coefficients = pad_list(coefficients, target_length=n_padded_terms, pad_value=0.0)
        exponents = pad_list(exponents, target_length=n_padded_terms, pad_value=PolyInts(0))

        muscle_function_path_data.append(
            MuscleFunctionPathData(
                name=muscle_path_name,
                coordinates=coordinates,
                coefficients=coefficients,
                dimension=dimension,
                order=order,

                num_tiles=n_tiles,
                exponents=exponents,
            )
        )
    return muscle_function_path_data


def path_type_to_muscle(
        muscle_function_paths: list[MuscleFunctionPathData]
) -> tuple[list[int], list[int], list[int]]:
    """
    Get mapping from (point | function | function_tiled) id to muscle id
    This function decides what type of path each muscle should use
    """
    point_paths, function_paths, function_tiled_paths = [], [], []
    for i, muscle_path in enumerate(muscle_function_paths):
        if muscle_path == USE_POINT_PATH:
            point_paths.append(i)
        else:
            function_paths.append(i)

        # TODO: tiles just aren't that fast, maybe future work
        # else:  # We need tiles if max number of dofs is higher. Performance is probably also a bit better for these
        #     function_tiled_paths.append(i)

    return point_paths, function_paths, function_tiled_paths


def get_fn_path_term_coeffs(muscle_function_paths: list[MuscleFunctionPathData]) -> list[float]:
    """ Gets the coefficients of the polynomial terms for all the muscle function paths """
    coeffs = []
    for muscle_path in muscle_function_paths:
        coeffs.extend(muscle_path.coefficients)
    return coeffs


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


def get_fn_path_dimension(muscle_function_paths: list[MuscleFunctionPathData]) -> list[int]:
    """ Gets the dimension of each muscle function path """
    return [muscle_path.dimension for muscle_path in muscle_function_paths]


def get_fn_path_order(muscle_function_paths: list[MuscleFunctionPathData]) -> list[int]:
    return [muscle_path.order for muscle_path in muscle_function_paths]


# --- TILED MUSCLE FUNCTION PATHS ---
def compute_num_function_tiles(
        muscle_function_paths: list[MuscleFunctionPathData],
        function_tiled_paths_id: list[int]
) -> int:
    """ Computes the total number of tiles needed to process all muscle function paths """
    num_total_tiles = 0
    for muscle_id in function_tiled_paths_id:
        muscle_path = muscle_function_paths[muscle_id]
        num_total_tiles += muscle_path.num_tiles
    return num_total_tiles


def get_fn_path_term_exps(muscle_function_paths: list[MuscleFunctionPathData]) -> list[PolyInts]:
    """ Gets the exponents of the polynomial terms for all the muscle function paths """
    exps = []
    for muscle_path in muscle_function_paths:
        exps.extend(muscle_path.exponents)
    return exps


def get_fn_tile_muscle_id(
        muscle_function_paths: list[MuscleFunctionPathData],
        function_tiled_paths_id: list[int]
) -> list[int]:
    """ Get the muscle id for each tile """
    tile_muscle_id = []
    for muscle_id in function_tiled_paths_id:
        muscle_path = muscle_function_paths[muscle_id]
        tile_muscle_id.extend([muscle_id] * muscle_path.num_tiles)
    return tile_muscle_id


def compute_fn_tile_offset(
        muscle_function_paths: list[MuscleFunctionPathData],
        function_tiled_paths_id: list[int]
) -> list[int]:
    """ Get the offset within the muscle function path for each tile """
    tile_offset = []
    for muscle_id in function_tiled_paths_id:
        muscle_path = muscle_function_paths[muscle_id]
        tile_offset.extend([i for i in range(muscle_path.num_tiles)])
    return tile_offset
