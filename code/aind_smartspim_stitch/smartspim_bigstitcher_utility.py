import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# Pasting the code here we need
def parse_json(json_path: str, s3_data_path: str, microns=False) -> ET.ElementTree:

    def extract_tile_channel_numbers(json_dict: dict):

        tile_channel_list = []
        for i, item in enumerate(json_dict):
            # get the channel number
            channel_number = get_tile_channel(item["file"])
            # print(f'channel_number: {channel_number}')
            tile_channel_list.append(channel_number)
        return tile_channel_list

    def get_tile_channel(tile_name: str):
        """Extracts the channel number from the tile_name"""

        # get the channel number only if the tile name has 'ch' in it
        if "ch" in tile_name:
            print(f"tile_name: {tile_name}")
            channel_number = tile_name.split("_")[-1].split(".")[0]
        else:
            channel_number = 0  # assume we are doing single channel stitching
        return channel_number

    def get_tile_number_lookup(json_dict: dict):
        # TODO rewrite this to make only a minimal subset of tiles, so that channels that share a tile are not duplicated

        # json_data = json.load(json_file.open('r'))

        # get filenames
        filename_numbers = [Path(item["file"]).stem for item in json_dict]

        # make a dictionary for the lookup
        tile_number_lookup = {}
        for i, item in enumerate(filename_numbers):
            tile_number_lookup[item] = i

        return tile_number_lookup

    def tile_number_to_position(tile_number: int, json_dict: dict):
        """Converts tile number to x, y, z position"""
        # load the json file
        # json_data = json.load(json_file.open('r'))

        # get min and max positions for XYZ
        position_list = [item["position"] for item in json_dict]

        x_position_list = [item[0] for item in position_list]
        y_position_list = [item[1] for item in position_list]
        # z_position_list = [item[2] for item in position_list]

        # find first non zero value in sorted list
        sorted_x_position_list = sorted(np.unique(x_position_list))
        x_min = sorted_x_position_list[1]

        sorted_y_position_list = sorted(np.unique(y_position_list))
        y_min = sorted_y_position_list[1]

        # sorted_z_position_list = sorted(np.unique(z_position_list))
        # z_min = sorted_z_position_list[1]

        tile_x_pos = []
        tile_y_pos = []
        tile_z_pos = []

        for i, item in enumerate(position_list):

            tile_x_pos.append(np.round(item[0] / x_min))
            tile_y_pos.append(np.round(item[1] / y_min))
            tile_z_pos.append(0)

        # print(f'tile_number: {tile_number}')
        x_pos = int(tile_x_pos[tile_number])
        y_pos = int(tile_y_pos[tile_number])
        z_pos = int(tile_z_pos[tile_number])
        # print(f'x_pos: {x_pos}, y_pos: {y_pos}, z_pos: {z_pos}')
        return x_pos, y_pos, z_pos

    def extract_tile_names(json_dict: dict) -> list[str]:

        round = Path(json_dict[0]["file"]).parent.stem[-1]

        tile_number_lookup = get_tile_number_lookup(json_dict)

        tile_names: list[str] = []
        # tile_numbers: list[str] = []
        for row in json_dict:
            # Ex: TILE_XXXX_ch_0.zarr
            tile_number = Path(row["file"]).stem
            tile_number = tile_number_lookup[tile_number]
            X, Y, Z = tile_number_to_position(int(tile_number), json_dict)
            # tile_name = f'TILE_{Path(row["file"]).stem}_ch_0.zarr'
            tile_name = f"R{round}_X_{X:04d}_Y_{Y:04d}_Z_{Z:04d}_ch_.zarr"
            tile_names.append(tile_name)
        return tile_names

    def extract_tile_names_unaltered(json_dict: dict) -> list[str]:

        tile_names: list[str] = []
        for row in json_dict:
            # Ex: TILE_XXXX_ch_0.zarr
            tile_name = Path(row["file"]).stem
            tile_name = f"{tile_name}.zarr"
            tile_names.append(tile_name)
        return tile_names

    def extract_tile_sizes(json_dict: dict) -> list[list[int]]:

        tile_sizes: list[list[int]] = []
        for row in json_dict:
            tile_sizes.append(row["size"])
        return tile_sizes

    def extract_tile_resolution(json_dict: dict, microns=False) -> list[float]:

        try:
            return json_dict[0]["pixelResolution"]
        except:
            pixel_resolution_list_m = json_dict[0]["pixel_resolution"]

            pixel_resolution_um = pixel_resolution_list_m.copy()
            if not microns:
                pixel_resolution_um = [str(float(i) * 1e6) for i in pixel_resolution_list_m]

            return pixel_resolution_um

    def extract_tile_translations(json_dict: dict) -> list[list[float]]:

        tile_transforms: list[list[float]] = []
        for row in json_dict:
            # Ex: [1851, 3703, 0]
            translation = row["position"]
            tile_transforms.append(translation)

        return tile_transforms

    def add_sequence_description(
        parent: ET.Element,
        tiles: list[str],
        tile_sizes: list[list[int]],
        tile_resolution: list[float],
        tile_channel_number: list[int],
        s3_data_path: str,
    ) -> None:
        def add_image_loader(
            seq_desc: ET.Element,
            tiles: list[str],
            s3_data_path: str,
        ) -> None:
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

        def add_view_setups(
            seq_desc: ET.Element,
            tiles: list[str],
            tile_sizes: list[list[int]],
            tile_resolution: list[float],
            tile_channel_number: list[int],
        ) -> None:
            def add_attributes(
                view_setups: ET.Element,
                unique_channel_list: list[int],
            ) -> None:
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
                    id_entry.text = (
                        f"{i}"  # this was only reporting 0 last time... should be incrementing
                    )
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
            add_attributes(view_setups, unique_channel_list)

        def add_time_points(seq_desc: ET.Element) -> None:
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

    # Gather info from json
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    # sort dictionary by tile name
    json_dict.sort(key=lambda e: e["file"])
    # tile_names = extract_tile_names(json_dict)
    tile_names = extract_tile_names_unaltered(json_dict)
    tile_sizes = extract_tile_sizes(json_dict)
    tile_resolution = extract_tile_resolution(json_dict, microns=microns)
    tile_translations = extract_tile_translations(json_dict)
    tile_channel_numbers = extract_tile_channel_numbers(json_dict)

    # Construct the output xml
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
    ET.indent(tree, space="\t", level=0)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def convert_json_to_xml(json_loc, s3_data_path, xml_name=None):

    tree = parse_json(json_loc, s3_data_path)

    dataset_to_process = Path(json_loc).parent.as_posix()
    if xml_name is not None:
        write_xml(tree, dataset_to_process + "/" + xml_name + ".xml")
    else:
        write_xml(tree, dataset_to_process + "/stitching_spot_channels.xml")
