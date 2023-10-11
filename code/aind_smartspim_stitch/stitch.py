"""
This file controls the alignment step
for a SmartSPIM dataset using TeraStitcher
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from aind_data_schema.processing import DataProcess, ProcessName

from .__init__ import __version__
from ._shared.types import PathLike
from .utils import utils


def build_parallel_command(params: dict, tool: PathLike) -> str:
    """
    Builds a mpi command based on a provided configuration dictionary.

    Parameters
    ------------------------
    params: dict
        Configuration dictionary used to build
        the mpi command depending on the platform.

    tool: PathLike
        Parallel tool to be used in the command.
        (Parastitcher or Paraconverter)

    Returns
    ------------------------
    str:
        Command that will be executed for terastitcher.

    """

    cpu_params = params["cpu_params"]

    # mpiexec for windows, mpirun for linux or macs OS
    mpi_command = "mpirun -np"
    additional_params = ""
    hostfile = ""
    n_procs = cpu_params["number_processes"]

    # Additional params provided in the configuration
    if len(cpu_params["additional_params"]):
        additional_params = utils.helper_additional_params_command(cpu_params["additional_params"])

    hostfile = f"--hostfile {cpu_params['hostfile']}"

    cmd = f"{mpi_command} {n_procs} {hostfile} {additional_params}"
    cmd += f"python {tool}"
    return cmd


def terastitcher_import_cmd(
    input_path: PathLike,
    xml_output_path: PathLike,
    import_params: dict,
    channel_name: str,
) -> str:
    """
    Builds the terastitcher's import command based on
    a provided configuration dictionary. It outputs
    a json file in the xmls folder of the output
    directory with all the parameters
    used in this step.

    Parameters
    ------------------------
    import_params: dict
        Configuration dictionary used to build the
        terastitcher's import command.

    channel_name:str
        Name of the dataset channel that will be imported

    fuse_path:PathLike
        Path where the fused xml files will be stored.
        This will only be used in multichannel fusing.
        Default None

    Returns
    ------------------------
    Tuple[str, str]:
        Command that will be executed for terastitcher and
        the TeraStitcher import binary
    """

    xml_output_path = Path(xml_output_path)

    volume_input = f"--volin={input_path}"

    output_path = xml_output_path.joinpath(f"xml_import_{channel_name}.xml")

    import_params["mdata_bin"] = str(xml_output_path.joinpath(f"mdata_{channel_name}.bin"))

    output_folder = f"--projout={output_path}"

    parameters = utils.helper_build_param_value_command(import_params)

    additional_params = ""
    if len(import_params["additional_params"]):
        additional_params = utils.helper_additional_params_command(import_params["additional_params"])

    cmd = f"terastitcher --import {volume_input} {output_folder} {parameters} {additional_params}"

    output_json = xml_output_path.joinpath(f"import_params_{channel_name}.json")
    utils.save_dict_as_json(f"{output_json}", import_params, True)

    return cmd, import_params["mdata_bin"]


def terastitcher_align_cmd(
    metadata_folder: PathLike, align_params: dict, channel_name: str, parastitcher_path: PathLike
) -> str:
    """
    Builds the terastitcher's align command based on
    a provided configuration dictionary. It outputs a
    json file in the xmls folder of the output directory
    with all the parameters used in this step.

    Parameters
    ------------------------
    params: dict
        Configuration dictionary used to
        build the terastitcher's align command.
    channel:str
        Name of the dataset channel that will be aligned

    Returns
    ------------------------
    str:
        Command that will be executed for terastitcher.

    """
    metadata_folder = Path(metadata_folder)

    input_path = metadata_folder.joinpath(f"xml_import_{channel_name}.xml")
    input_xml = f"--projin={input_path}"

    output_path = metadata_folder.joinpath(f"xml_displcomp_{channel_name}.xml")
    output_xml = f"--projout={output_path}"

    parallel_command = build_parallel_command(align_params, "align", parastitcher_path)

    parameters = utils.helper_build_param_value_command(align_params)

    cmd = f"{parallel_command} --displcompute {input_xml} {output_xml} {parameters} > {metadata_folder}/align_step.txt"

    output_json = metadata_folder.joinpath(f"align_params_{channel_name}.json")
    utils.save_dict_as_json(f"{output_json}", align_params, True)

    return cmd


def terastitcher_no_params_steps_cmd(
    metadata_folder: PathLike,
    step_name: str,
    input_xml: str,
    output_xml: str,
    channel_name: str,
    io_params: Optional[dict] = None,
) -> str:
    """
    Builds the terastitcher's input-output commands
    based on a provided configuration dictionary.
    These commands are: displproj for projection,
    displthres for threshold and placetiles for placing tiles.
    Additionally, it outputs a json file in the xmls folder
    of the output directory with all the parameters used
    in this step.

    Parameters
    ------------------------
    metadata_folder: PathLike
        Path where the metadata for the XMLs is being
        stored

    step_name: str
        Name of the step that will be executed.
        The names should be: 'displproj' for projection,
        'displthres' for threshold and 'placetiles'
        for placing tiles step.

    input_xml: str
        The xml filename outputed from the previous command.

    output_xml: str
        The xml filename that will be used as output for this step.

    io_params: Optional[dict]
        Configuration dictionary used to build the terastitcher's command.

    Returns
    ------------------------
    str:
        Command that will be executed for terastitcher.

    """
    metadata_folder = Path(metadata_folder)

    input_xml = f"--projin={metadata_folder.joinpath(input_xml)}"

    output_xml = f"--projout={metadata_folder.joinpath(output_xml)}"

    parameters = ""

    if io_params:
        parameters = utils.helper_build_param_value_command(io_params)

    cmd = f"terastitcher --{step_name} {input_xml} {output_xml} {parameters}"

    output_json = metadata_folder.joinpath(f"{step_name}_params_{channel_name}.json")
    utils.save_dict_as_json(f"{output_json}", io_params, True)

    return cmd


def terastitcher_stitch(
    data_folder: PathLike,
    metadata_folder: PathLike,
    channel_name: str,
    smartspim_config: dict,
    logger: logging.Logger,
    code_url: Optional[str] = "https://github.com/AllenNeuralDynamics/aind-smartspim-stitch",
) -> Tuple:
    """
    Creates the alignment XML for a single
    channel using TeraStitcher.

    Parameters
    ----------
    data_folder: PathLike
        Path where the data is located

    metadata_folder: PathLike
        Path where the XMLs and parameters
        will be saved

    channel_name: str
        Channel name to stitch

    smartspim_config: dict
        Dictionary with the all the information
        to stitch a smartspim dataset

    logger: logging.Logger
        Logger to print messages

    code_url: Optional[str]
        Github repository where this code is
        hosted to include in the metadata

    Returns
    -------
    Tuple[PathLike, List[DataProcess]]
        Tuple with the path where the
        optimized transformations are saved
        in XML format and the data processes
        metadata
    """

    data_processes = []

    parastitcher_path = Path(smartspim_config["pyscripts_path"]).joinpath("Parastitcher.py")
    paraconverter_path = Path(smartspim_config["pyscripts_path"]).joinpath("paraconverter.py")

    # Converting to Path object
    data_folder = Path(data_folder)
    metadata_folder = Path(metadata_folder)

    channel_path = data_folder.joinpath(channel_name)

    if not channel_path.exists():
        raise FileExistsError(f"Path {channel_path} does not exist!")

    logger.info(f"Starting importing for channel {channel_name}")

    teras_import_channel_cmd, teras_import_binary = terastitcher_import_cmd(
        input_path=channel_path,
        xml_output_path=metadata_folder,
        import_params=smartspim_config["import_data"],
        channel_name=channel_name,
    )

    logger.info(f"Executing TeraStitcher command: {teras_import_channel_cmd}")

    # Importing channel to generate binary file
    import_start_time = datetime.now()
    utils.execute_command(command=teras_import_channel_cmd, logger=logger, verbose=True)
    import_end_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_IMPORTING,
            software_version="1.11.10",
            start_date_time=import_start_time,
            end_date_time=import_end_time,
            input_location=str(channel_path),
            output_location=str(metadata_folder),
            code_url=code_url,
            code_version=__version__,
            parameters=smartspim_config["import_data"],
            notes=f"TeraStitcher image import for channel {channel_name}",
        )
    )

    logger.info(f"Starting alignment for channel {channel_name}")

    teras_align_channel_cmd = terastitcher_align_cmd(
        metadata_folder=metadata_folder,
        align_params=smartspim_config["align"],
        channel_name=channel_name,
        parastitcher_path=parastitcher_path,
    )

    logger.info(f"Executing TeraStitcher command: {teras_align_channel_cmd}")

    # Aligning channel to generate binary file
    align_start_time = datetime.now()
    utils.execute_command(command=teras_align_channel_cmd, logger=logger, verbose=True)
    align_end_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_TILE_ALIGNMENT,
            software_version="1.11.10",
            start_date_time=align_start_time,
            end_date_time=align_end_time,
            input_location=str(metadata_folder.joinpath(f"xml_import_{channel_name}.xml")),
            output_location=str(metadata_folder.joinpath(f"xml_displcomp_{channel_name}.xml")),
            code_url=code_url,
            code_version=__version__,
            parameters=smartspim_config["align"],
            notes=f"TeraStitcher image alignment for channel {channel_name} using NCC algorithm",
        )
    )

    logger.info(f"Starting projection for channel {channel_name}")

    teras_projection_cmd = terastitcher_no_params_steps_cmd(
        metadata_folder=metadata_folder,
        step_name="displproj",
        input_xml=f"xml_displcomp_{channel_name}.xml",
        output_xml=f"xml_displproj_{channel_name}.xml",
        channel_name=channel_name,
        io_params=None,
    )

    logger.info(f"Executing TeraStitcher command: {teras_projection_cmd}")

    projection_start_time = datetime.now()
    utils.execute_command(command=teras_projection_cmd, logger=logger, verbose=True)
    projection_end_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_TILE_PROJECTION,
            software_version="1.11.10",
            start_date_time=projection_start_time,
            end_date_time=projection_end_time,
            input_location=str(metadata_folder.joinpath(f"xml_displcomp_{channel_name}.xml")),
            output_location=str(metadata_folder.joinpath(f"xml_displproj_{channel_name}.xml")),
            code_url=code_url,
            code_version=__version__,
            parameters={},
            notes=f"Projection in channel {channel_name}",
        )
    )

    logger.info(f"Starting thresholding for channel {channel_name}")

    threshold_cnf = {"threshold": smartspim_config["threshold"]["reliability_threshold"]}

    teras_thres_cmd = terastitcher_no_params_steps_cmd(
        metadata_folder=metadata_folder,
        step_name="displthres",
        input_xml=f"xml_displproj_{channel_name}.xml",
        output_xml=f"xml_displthres_{channel_name}.xml",
        channel_name=channel_name,
        io_params=threshold_cnf,
    )

    thres_start_time = datetime.now()
    utils.execute_command(command=teras_thres_cmd, logger=logger, verbose=True)
    thres_end_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_THRESHOLDING,  # "Image thresholding"
            software_version="1.11.10",
            start_date_time=thres_start_time,
            end_date_time=thres_end_time,
            input_location=str(metadata_folder.joinpath(f"xml_displproj_{channel_name}.xml")),
            output_location=str(metadata_folder.joinpath(f"xml_displthres_{channel_name}.xml")),
            code_url=code_url,
            code_version=__version__,
            parameters=smartspim_config["threshold"],
            notes=f"TeraStitcher thresholding in channel {channel_name}",
        )
    )

    merge_xml_informative = f"{metadata_folder}/xml_merging_{channel_name}.xml"
    logger.info(f"Starting placing tiles for channel {channel_name}")

    teras_placing_cmd = terastitcher_no_params_steps_cmd(
        metadata_folder=metadata_folder,
        step_name="placetiles",
        input_xml=f"xml_displthres_{channel_name}.xml",
        output_xml=merge_xml_informative,
        channel_name=channel_name,
        io_params=None,
    )

    placing_start_time = datetime.now()
    utils.execute_command(command=teras_placing_cmd, logger=logger, verbose=True)
    placing_end_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_TILE_ALIGNMENT,  # "Image tile alignment"
            software_version="1.11.10",
            start_date_time=placing_start_time,
            end_date_time=placing_end_time,
            input_location=str(metadata_folder.joinpath(f"xml_displthres_{channel_name}.xml")),
            output_location=merge_xml_informative,
            code_url=code_url,
            code_version=__version__,
            parameters={},
            notes=f"Global optimization using MST algorithm for channel {channel_name}",
        )
    )

    return merge_xml_informative, data_processes


def main(
    data_folder: PathLike,
    output_alignment_path: PathLike,
    smartspim_config: dict,
    channel_regex: Optional[str] = r"Ex_([0-9]*)_Em_([0-9]*)$",
    final_alignment_name: Optional[str] = "volume_alignments.xml",
):
    """
    This function fuses a SmartSPIM dataset.

    Parameters
    -----------
    data_folder: PathLike
        Path where the image data is located

    output_alignment_path: PathLike
        Path where the XML with the
        transformations will be saved

    smartspim_config: dict
        Dictionary with the smartspim configuration
        for that dataset

    channel_regex: Optional[str]
        Regular expression to identify smartspim channels

    final_alignment_name: Optional[str]
        Name of the file that will be used to copy
        the generated XML with the transformations
        to the output_alignment_path parameter.
        Ideally, the output_alignment_path will point
        to the results folder and then the filename
        will be f"{output_alignment_path}/{final_alignment_name}".

        This is done to make the pipeline easier to build
        in Code Ocean.

        Default="volume_alignments.xml"
    """

    # Converting to path objects if necessary
    data_folder = Path(data_folder)
    output_alignment_path = Path(output_alignment_path)

    if not output_alignment_path.exists():
        raise FileNotFoundError(f"XML path {output_alignment_path} does not exist")

    # Looking for SmartSPIM channels on data folder
    smartspim_channels = utils.find_smartspim_channels(path=data_folder, channel_regex=channel_regex)

    if not len(smartspim_channels):
        raise ValueError("No SmartSPIM channels found!")

    # Finding stitching channel in the found channels

    channel_name = None

    for smartspim_channel in smartspim_channels:
        if smartspim_config["stitching"]["channel"] in smartspim_channel:
            channel_name = smartspim_channels
            break

    if channel_name is None:
        raise ValueError(
            f"Channel {smartspim_config['stitching']['channel']} not found in {smartspim_channels}"
        )

    # Contains the paths where I'll place the
    # alignment metadata
    (metadata_folder,) = utils.create_align_folder_structure(
        output_alignment_path=output_alignment_path, channel_name=channel_name
    )

    # Logger pointing everything to the metadata path
    logger = utils.create_logger(output_log_path=metadata_folder)

    logger.info(f"Output folders - Stitch metadata: {metadata_folder}")

    logger.info(f"Generating derived data description")

    terastitcher_alignment_filepath, data_processes = terastitcher_stitch(
        data_folder=data_folder,
        metadata_folder=metadata_folder,
        channel_name=channel_name,
        smartspim_config=smartspim_config,
        logger=logger,
        channel_regex=channel_regex,
    )

    logger.info(f"Final alignment file with TeraStitcher in path: {terastitcher_alignment_filepath}")

    # Copying the alignment transforms to the output path
    utils.copy_file(
        input_filename=terastitcher_alignment_filepath,
        output_filename=output_alignment_path.joinpath(final_alignment_name),
    )

    # Generating independent processing json
    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=metadata_folder,
        processor_full_name="Camilo Laiton",
        pipeline_version="1.5.0",
    )
