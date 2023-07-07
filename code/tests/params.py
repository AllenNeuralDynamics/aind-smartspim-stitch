"""
Parameters for tests
"""


def get_save_dict_as_json_params():
    """
    Returns parameters for the
    save_dict_as_json function
    """
    return [[{"number": 13, "text": "this is a test"}, "test_json.json"]]


def get_read_json_as_dict_params():
    """
    Returns parameters for the
    read_json_as_dict function
    """
    return [
        ["test_json_2.json", {"number": 13, "text": "this is a test"}],
        ["path/does/not/exists.json", {}],
    ]


def get_helper_build_param_value_command_params():
    """
    Returns parameters for the
    helper_build_param_value_command function
    """
    return [
        [
            {
                "ref1": "X",
                "ref2": "Y",
                "ref3": "D",
                "vxl1": 1.8,
                "vxl2": 1.8,
                "vxl3": 2,
                "additional_params": ["sparse_data", "libtiff_uncompress"],
            },
            "--ref1=X --ref2=Y --ref3=D --vxl1=1.8 --vxl2=1.8 --vxl3=2 ",
        ],
        [
            {
                "ref1": "X",
                "ref2": "Y",
                "ref3": "D",
                "vxl1": 1.8,
                "vxl2": 1.8,
                "vxl3": 2,
                "additional_params": ["sparse_data", "libtiff_uncompress"],
            },
            "--ref1 X --ref2 Y --ref3 D --vxl1 1.8 --vxl2 1.8 --vxl3 2 ",
            False,
        ],
        [{}, ""],
        [{}, "", False],
    ]


def get_helper_additional_params_command_params():
    """
    Returns parameters for the
    helper_additional_params_command function
    """
    return [
        [["sparse_data", "libtiff_uncompress"], "--sparse_data --libtiff_uncompress ",],
        [[], ""],
    ]


# ZarrConverter tests params
def get_compute_pyramid_params():
    """
    Returns parameters for the
    compute_pyramid function
    """
    return [[5, [1, 2, 2, 2]], [1, [1, 2, 2, 2]]]


def get_compute_pyramid_raises_params():
    """
    Returns parameters for the
    compute_pyramid_raises function
    """
    return [[5, [1, 1, 2, 2, 2]], [5, []]]
