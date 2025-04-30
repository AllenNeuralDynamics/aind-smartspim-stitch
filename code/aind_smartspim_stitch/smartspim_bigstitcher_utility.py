"""
Smartspim BigStitcher Utility Module
This module provides functions to convert JSON metadata files into XML format
suitable for BigStitcher, a software for stitching large image datasets.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def get_tile_channel(tile_name: str) -> int:
    """
    Extracts the channel number from the tile name.

    Parameters
    ----------
    tile_name : str
        Name of the tile file.

    Returns
    -------
    int
        Channel number extracted from the tile name.
    """
    if "ch" in tile_name:
        channel_number = tile_name.split("_")[-1].split(".")[0]
    else:
        channel_number = 0  # Assume single-channel stitching
    return int(channel_number)


def get_tile_number_lookup(json_dict: dict) -> dict:
    """
    Creates a lookup dictionary for tile numbers based on filenames.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing tile metadata.

    Returns
    -------
    dict
        Dictionary mapping tile filenames to their indices.
    """
    filename_numbers = [Path(item["file"]).stem for item in json_dict]
    tile_number_lookup = {item: i for i, item in enumerate(filename_numbers)}
    return tile_number_lookup


def extract_tile_channel_numbers(json_dict: dict) -> list[int]:
    """
    Extracts channel numbers from the tile metadata.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing tile metadata.

    Returns
    -------
    list[int]
        List of channel numbers for each tile.
    """
    tile_channel_list = []
    for i, item in enumerate(json_dict):
        channel_number = get_tile_channel(item["file"])
        tile_channel_list.append(channel_number)
    return tile_channel_list


def tile_number_to_position(tile_number: int, json_dict: dict) -> tuple[int, int, int]:
    """
    Converts a tile number to its x, y, z position.

    Parameters
    ----------
    tile_number : int
        Tile number to convert.
    json_dict : dict
        Dictionary containing tile metadata.

    Returns
    -------
    tuple[int, int, int]
        x, y, z position of the tile.
    """
    position_list = [item["position"] for item in json_dict]
    x_position_list = [item[0] for item in position_list]
    y_position_list = [item[1] for item in position_list]

    sorted_x_position_list = sorted(np.unique(x_position_list))
    x_min = sorted_x_position_list[1]

    sorted_y_position_list = sorted(np.unique(y_position_list))
    y_min = sorted_y_position_list[1]

    tile_x_pos = [np.round(item[0] / x_min) for item in position_list]
    tile_y_pos = [np.round(item[1] / y_min) for item in position_list]
    tile_z_pos = [0 for _ in position_list]

    x_pos = int(tile_x_pos[tile_number])
    y_pos = int(tile_y_pos[tile_number])
    z_pos = int(tile_z_pos[tile_number])
    return x_pos, y_pos, z_pos


def extract_tile_names_unaltered(json_dict: dict) -> list[str]:
    """
    Extracts unaltered tile names from the metadata.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing tile metadata.

    Returns
    -------
    list[str]
        List of unaltered tile names.
    """
    return [f"{Path(row['file']).stem}.zarr" for row in json_dict]


def extract_tile_sizes(json_dict: dict) -> list[list[int]]:
    """
    Extracts tile sizes from the metadata.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing tile metadata.

    Returns
    -------
    list[list[int]]
        List of tile sizes (width, height, depth).
    """
    return [row["size"] for row in json_dict]


def extract_tile_resolution(json_dict: dict, microns=False) -> list[float]:
    """
    Extracts tile resolution from the metadata.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing tile metadata.
    microns : bool, optional
        Whether to return resolution in microns. Defaults to False.

    Returns
    -------
    list[float]
        Tile resolution as a list of floats.
    """
    try:
        return json_dict[0]["pixelResolution"]
    except KeyError:
        pixel_resolution_list_m = json_dict[0]["pixel_resolution"]
        if not microns:
            return [float(i) * 1e6 for i in pixel_resolution_list_m]
        return pixel_resolution_list_m


def extract_tile_translations(json_dict: dict) -> list[list[float]]:
    """
    Extracts tile translations from the metadata.

    Parameters
    ----------
    json_dict : dict
        Dictionary containing tile metadata.

    Returns
    -------
    list[list[float]]
        List of tile translations (x, y, z).
    """
    return [row["position"] for row in json_dict]


def add_image_loader(
    seq_desc: ET.Element,
    tiles: list[str],
    s3_data_path: str,
) -> None:
    """
    Adds an image loader element to the sequence description.

    Parameters
    ----------
    seq_desc : ET.Element
        The sequence description XML element.
    tiles : list[str]
        List of tile names.
    s3_data_path : str
        Path to the S3 bucket or local directory where the data is stored.

    """
    img_loader = ET.SubElement(seq_desc, "ImageLoader")
    img_loader.attrib["format"] = "bdv.multimg.zarr"
    img_loader.attrib["version"] = "1.0"
    x = ET.SubElement(img_loader, "zarr")
    x.attrib["type"] = "absolute"

    x.text = s3_data_path

    zgs = ET.SubElement(img_loader, "zgroups")
    for i, tile in enumerate(tiles):
        zg = ET.SubElement(zgs, "zgroup")
        zg.attrib["setup"] = f"{i}"
        zg.attrib["timepoint"] = "0"
        x = ET.SubElement(zg, "path")
        x.text = tile


def add_attributes(
    view_setups: ET.Element,
    unique_channel_list: list[int],
    tiles: list[str],
) -> None:
    """
    Adds attributes to the view setups.

    Parameters
    ----------
    view_setups : ET.Element
        The view setups XML element.
    unique_channel_list : list[int]
        List of unique channel numbers.
    tiles : list[str]
        List of tile names.

    """
    # Add attributes
    x = ET.SubElement(view_setups, "Attributes")
    x.attrib["name"] = "illumination"
    x = ET.SubElement(x, "Illumination")
    y = ET.SubElement(x, "id")
    y.text = "0"
    y = ET.SubElement(x, "name")
    y.text = "0"

    x = ET.SubElement(view_setups, "Attributes")
    x.attrib["name"] = "channel"

    for i, channel in enumerate(unique_channel_list):
        z = ET.SubElement(x, "Channel")
        y = ET.SubElement(z, "id")
        y.text = f"{channel}"  # should this be
        y = ET.SubElement(z, "name")
        y.text = f"{channel}"

    tile_atts = ET.SubElement(view_setups, "Attributes")
    tile_atts.attrib["name"] = "tile"
    for i, tile in enumerate(tiles):
        t_entry = ET.Element("Tile")
        id_entry = ET.Element("id")
        id_entry.text = f"{i}"  # this was only reporting 0 last time... should be incrementing
        t_entry.append(id_entry)
        name_entry = ET.Element("name")
        name_entry.text = str(tile)
        t_entry.append(name_entry)
        tile_atts.append(t_entry)

    x = ET.SubElement(view_setups, "Attributes")
    x.attrib["name"] = "angle"
    x = ET.SubElement(x, "Angle")
    y = ET.SubElement(x, "id")
    y.text = "0"
    y = ET.SubElement(x, "name")
    y.text = "0"


def add_view_setups(
    seq_desc: ET.Element,
    tiles: list[str],
    tile_sizes: list[list[int]],
    tile_resolution: list[float],
    tile_channel_number: list[int],
) -> None:
    """
    Adds view setups to the sequence description.

    Parameters
    ----------
    seq_desc : ET.Element
        The sequence description XML element.
    tiles : list[str]
        List of tile names.
    tile_sizes : list[list[int]]
        List of tile sizes, where each size is a list of [width, height, depth].
    tile_resolution : list[float]
        Resolution of the tiles in microns.
    tile_channel_number : list[int]
        List of channel numbers corresponding to each tile.

    """

    view_setups = ET.SubElement(seq_desc, "ViewSetups")
    tile_count = 0
    for i, (tile, t_size) in enumerate(zip(tiles, tile_sizes)):
        vs = ET.SubElement(view_setups, "ViewSetup")

        x = ET.SubElement(vs, "id")
        x.text = f"{i}"
        x = ET.SubElement(vs, "name")
        x.text = str(tile)
        x = ET.SubElement(vs, "size")
        x.text = f"{t_size[0]} {t_size[1]} {t_size[2]}"

        voxel_size = ET.SubElement(vs, "voxelSize")
        x = ET.SubElement(voxel_size, "unit")
        x.text = "µm"
        x = ET.SubElement(voxel_size, "size")
        x.text = f"{tile_resolution[0]} {tile_resolution[1]} {tile_resolution[2]}"

        attr = ET.SubElement(vs, "attributes")
        x = ET.SubElement(attr, "illumination")
        x.text = "0"
        x = ET.SubElement(attr, "channel")  # add channel information
        x.text = f"{tile_channel_number[i]}"  # "0"
        x = ET.SubElement(attr, "tile")
        tile_count += 1
        x.text = f"{tile_count}"
        x = ET.SubElement(attr, "angle")
        x.text = "0"  # No deskewing

    unique_channel_list = list(set(tile_channel_number))
    add_attributes(view_setups, unique_channel_list, tiles)


def add_sequence_description(
    parent: ET.Element,
    tiles: list[str],
    tile_sizes: list[list[int]],
    tile_resolution: list[float],
    tile_channel_number: list[int],
    s3_data_path: str,
) -> None:
    """
    Adds a sequence description to the XML structure.

    Parameters
    ----------
    parent : ET.Element
        The parent XML element to which the sequence description will be added.
    tiles : list[str]
        List of tile names.
    tile_sizes : list[list[int]]
        List of tile sizes, where each size is a list of [width, height, depth].
    tile_resolution : list[float]
        Resolution of the tiles in microns.
    tile_channel_number : list[int]
        List of channel numbers corresponding to each tile.
    s3_data_path : str
        Path to the S3 bucket or local directory where the data is stored.

    """

    def add_time_points(seq_desc: ET.Element) -> None:
        """
        Adds time points to the sequence description.

        Parameters
        ----------
        seq_desc : ET.Element
            The sequence description XML element.

        """
        x = ET.SubElement(seq_desc, "Timepoints")
        x.attrib["type"] = "pattern"
        y = ET.SubElement(x, "integerpattern")
        y.text = "0"
        ET.SubElement(seq_desc, "MissingViews")

    seq_desc = ET.SubElement(parent, "SequenceDescription")
    add_image_loader(seq_desc, tiles, s3_data_path)
    add_view_setups(seq_desc, tiles, tile_sizes, tile_resolution, tile_channel_number)
    add_time_points(seq_desc)


def add_view_registrations(parent: ET.Element, translations: list[list[float]]) -> None:
    """
    Adds view registrations to the XML structure.

    Parameters
    ----------
    parent : ET.Element
        The parent XML element to which the view registrations will be added.
    translations : list[list[float]]
        List of translations for each tile, where each translation is a list of [x, y, z].

    """
    view_registrations = ET.SubElement(parent, "ViewRegistrations")
    for i, tr in enumerate(translations):
        vr = ET.SubElement(view_registrations, "ViewRegistration")
        vr.attrib["timepoint"] = "0"
        vr.attrib["setup"] = f"{i}"

        vt = ET.Element("ViewTransform")
        vt.attrib["type"] = "affine"
        name = ET.SubElement(vt, "Name")
        name.text = "Translation to Nominal Grid"
        affine = ET.SubElement(vt, "affine")

        affine.text = (
            f"1.0 0.0 0.0 {str(float(tr[0]))} "
            + f"0.0 1.0 0.0 {str(float(tr[1]))} "
            + f"0.0 0.0 1.0 {str(float(tr[2]))}"
        )

        vr.append(vt)


def parse_json(json_path: str, s3_data_path: str, microns=False) -> ET.ElementTree:
    """
    Parses a JSON file and converts it into an XML ElementTree for BigStitcher.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing tile metadata.
    s3_data_path : str
        Path to the S3 bucket or local directory where the data is stored.
    microns : bool, optional
        Whether to return pixel resolution in microns. Defaults to False.

    Returns
    -------
    ET.ElementTree
        An XML ElementTree representing the parsed data.
    """
    # Nested helper functions are documented inline for clarity.

    # Main logic of the function
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    json_dict.sort(key=lambda e: e["file"])
    tile_names = extract_tile_names_unaltered(json_dict)
    tile_sizes = extract_tile_sizes(json_dict)
    tile_resolution = extract_tile_resolution(json_dict, microns=microns)
    tile_translations = extract_tile_translations(json_dict)
    tile_channel_numbers = extract_tile_channel_numbers(json_dict)

    spim_data = ET.Element("SpimData")
    spim_data.attrib["version"] = "0.2"
    x = ET.SubElement(spim_data, "BasePath")
    x.attrib["type"] = "relative"
    x.text = "."

    add_sequence_description(
        spim_data, tile_names, tile_sizes, tile_resolution, tile_channel_numbers, s3_data_path
    )
    add_view_registrations(spim_data, tile_translations)

    return ET.ElementTree(spim_data)


def write_xml(tree: ET.ElementTree, path: str) -> None:
    """
    Writes an XML ElementTree to a file.

    Parameters
    ----------
    tree : ET.ElementTree
        The XML tree to write.
    path : str
        Path to the output XML file.

    Returns
    -------
    None
    """
    ET.indent(tree, space="\t", level=0)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def convert_json_to_xml(json_loc: str, s3_data_path: str, xml_name: str = None) -> None:
    """
    Converts a JSON file to an XML file for BigStitcher.

    Parameters
    ----------
    json_loc : str
        Path to the input JSON file.
    s3_data_path : str
        Path to the S3 bucket or local directory where the data is stored.
    xml_name : str, optional
        Name of the output XML file. If None, a default name is used.

    Returns
    -------
    None
    """
    tree = parse_json(json_loc, s3_data_path)
    dataset_to_process = Path(json_loc).parent.as_posix()
    if xml_name is not None:
        write_xml(tree, dataset_to_process + "/" + xml_name + ".xml")
    else:
        write_xml(tree, dataset_to_process + "/stitching_spot_channels.xml")
