import os
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fiona
import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100


cur_dir = "/work/pi_smaji_umass_edu/rdaroya/landsat-eda"
DATA_DIR = f"{cur_dir}/data"
TO_GET_ALL_PREDS = False # If True, it will show all predictions that lie within the tile;
                        # If False, it will only show predictions that correspond to the exact landsat_id specified
TMP_DIR = f"{DATA_DIR}/tmp" # Directory where temporary images will be saved (will be removed later)
OUT_DIR = f"{cur_dir}/output"

VAL_FILE = "validation_v1.csv"
SHP_FP = "CONUS_nhd_reach_poly_collapse.shp"
LABEL_COLORS = ['red', 'green', 'blue', 'orange', 'magenta']
# CROPPED_TILE_SIDE_LEN = 200  # size in pixels of a side of a cropped image (smallest)
CROPPED_TILE_SIDE_LEN = 400  # size in pixels of a side of a cropped image
PCT_FRAME_SAVE = 0.3    #percentage of frame that should have river pixels to be saved


# Common landsat band combinations from https://www.usgs.gov/media/images/common-landsat-band-combinations
# BANDS = ["B7", "B5", "B4"]  # shortwave infrared
# BANDS = ["B6", "B5", "B4"]  # vegetative analysis
# BANDS = ["B7", "B6", "B4"]  # false color (urban)
# BANDS = ["B5", "B4", "B3"]  # color infrared
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]  # all bands
# BANDS = ["B7","B8"]  # all bands
BANDS_ID = "".join(BANDS).lower()   # for filename reference
IS_SEPARATE_BANDS = True       # Is set to True, will save bands separately

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if IS_SEPARATE_BANDS:
    plt.set_cmap("gray")

def normalize(img_array):
    """Normalize in the range of 0-1"""
    im_min, im_max = np.min(img_array), np.max(img_array)
    return (img_array-im_min)/(im_max - im_min)


def save_img(img, landsat_id, band_name = None):
    ft_text = ""
    logger.debug(f"Saving image for {landsat_id}")
    ax = plt.imshow(img, interpolation='none')
    for idx, (y,x) in enumerate(yxmarks):
        cur_color = LABEL_COLORS[idx%len(LABEL_COLORS)]
        plt.scatter(x, y, s=500, c=cur_color, marker='x')
        if labels is not None:
            ft_text += f"{cur_color}: GT:{labels[idx][0]}, P: {labels[idx][1]:.2f}, date:{dates[idx]}\n"
    # if dates is not None:
    #     title_str = f"Date: {dates}"
    # plt.title(title_str)
    plt.figtext(0.2, -0.1, ft_text, horizontalalignment='left', verticalalignment='bottom')
    if band_name is not None:   # means multiple bands were combined
        plt.savefig(f"{OUT_DIR}/{landsat_id}/_{landsat_id}_{BANDS_ID}.png", bbox_inches="tight")
    else:
        plt.savefig(f"{OUT_DIR}/{landsat_id}/_{landsat_id}_{band_name}.png", bbox_inches="tight")
    plt.show()
    plt.close()


def display_and_save_rgb_images(landsat_id, yxmarks=None, labels=None, dates=None, bands=BANDS):
    """
    Arguments:
    bands: list of band postfixes that end with ".tif"; should be ordered the way they should be displayed if combined
        for reference, blue=b2 green=b3 red=b4
    """
    band_imgs = {b:None for b in bands}
    band_fps = {b:None for b in bands}
    for _,_,files in os.walk(os.path.join(DATA_DIR,landsat_id)):
        for file in files:
            for b in bands:
                if file.lower().endswith(f"{b}.tif".lower()):
                    img_fp = os.path.join(DATA_DIR, landsat_id, file)
                    band_fps[b] = img_fp
                    img = rasterio.open(img_fp)
                    img_norm = normalize(np.array(img.read(1)))
                    if not IS_SEPARATE_BANDS:
                        band_imgs[b] = np.expand_dims(img_norm,axis=-1)
                    else:
                        save_img(img_norm, landsat_id, band_name=b)
        if not IS_SEPARATE_BANDS:
            stacked_img = np.concatenate([band_imgs[b] for b in bands], axis=2)
            save_img(stacked_img, landsat_id)


def save_raster_combined(landsat_id, file_list): # file_list is rgb order
    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(file_list))

    # Read each layer and write it to stack
    out_fp = f"{OUT_DIR}/{landsat_id}.TIF"
    with rasterio.open(out_fp, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def reproject_crs_to_files(landsat_dir, dst_crs = 'EPSG:4326', bands=BANDS):
    dst_fps = []
    for b in bands:
        with rasterio.open(f'{DATA_DIR}/{landsat_dir}/{landsat_dir}_{b}.TIF') as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            # TODO: Find a way to do this without having to save to a separate file
            dst_fp = f'{TMP_DIR}/{landsat_dir}_{b}_{dst_crs.split(":")[-1]}.TIF'
            dst = rasterio.open(dst_fp, 'w', **kwargs)
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

            dst_fps.append(dst_fp)
    save_raster_combined(landsat_id=landsat_dir, file_list=dst_fps)
    return dst_fps

def get_idx(data, lon, lat):
    """
    Based on the given latitude and longitude, get the x and y indices in the image
    Longitude is the x axis; Latitude is the y axis
    """
    # Based on https://gis.stackexchange.com/a/428784
    idx = data.index(lon, lat)
    return idx

def get_tile_metadata(val_data, landsat_id):
    # Reproject to correct CRS to get lat/lon equivalent
    dst_fps = reproject_crs_to_files(landsat_dir=landsat_id, dst_crs = 'EPSG:4326')

    # Find the lat/lon bounds of the tile
    data = rasterio.open(dst_fps[0])    # open the one geotiff that was converted to the lat/lon coordinate system
    min_lon = data.bounds.left
    max_lon = data.bounds.right
    min_lat = data.bounds.bottom
    max_lat = data.bounds.top

    # Find all data points in feature file that fall within the tile and fit conditions below
    if TO_GET_ALL_PREDS:
        bool_filter = (val_data["long"]>=min_lon) & (val_data["long"]<=max_lon) & (val_data["lat"]<= max_lat) & (val_data["lat"]>= min_lat)
    else:
        bool_filter = (val_data["landsat_id"] == landsat_id)
    filtered_df = val_data[bool_filter]

    # Filter datapoints and retrieve metadata
    dates, datapoints = [], []
    for idx, row in filtered_df.iterrows():
        y,x = get_idx(data, lon=row["long"], lat=row["lat"])
        datapoint = [int(y), int(x), row["Actual"], row["Predicted"], row["long"], row["lat"]]
        dates.append(str(row["date_unity"]))
        datapoints.append(datapoint)
    
    datapoints = np.array(datapoints)
    yxmarks = datapoints[:,0:2]
    labels = datapoints[:,2:4]
    long_lat = datapoints[:,4:]
    logger.debug(f"yxmarks.shape: {yxmarks.shape}")
    logger.debug(f"labels: {labels}")
    logger.debug(f"long_lat: {long_lat}")
    return yxmarks, labels, dates


def plot_metadata(plt, yxmarks, labels, dates):
    ft_text = ""
    if len(yxmarks):
        for idx, (y,x) in enumerate(yxmarks):
            cur_color = LABEL_COLORS[idx%len(LABEL_COLORS)]
            plt.scatter(x, y, s=500, c=cur_color, marker='x')
            if len(labels):
                ft_text += f"{cur_color}: GT:{labels[idx][0]}, P: {labels[idx][1]:.2f}, date:{dates[idx]}\n"
    plt.figtext(0.2, -0.1, ft_text, horizontalalignment='left', verticalalignment='bottom')

def plot_metadata_cropped(plt, yxmarks, labels, dates, bottom_right_yx, top_left_yx):
    is_marked = False
    ft_text = ""
    for idx, (y,x) in enumerate(yxmarks):
        cur_color = LABEL_COLORS[idx%len(LABEL_COLORS)]
        if x>=top_left_yx[1] and x<=bottom_right_yx[1] and y>=top_left_yx[0] and y<=bottom_right_yx[0]:
            plt.scatter(x-top_left_yx[1], y-top_left_yx[0], s=500, c=cur_color, marker='x')
            is_marked = True
        if len(labels):
            ft_text += f"{cur_color}: GT:{labels[idx][0]}, P: {labels[idx][1]:.2f}, date:{dates[idx]}\n"
    plt.figtext(0.2, -0.1, ft_text, horizontalalignment='left', verticalalignment='bottom')
    return is_marked

def get_masked_raster_bands(landsat_id, shapes, bands=BANDS):
    mask_imgs = []
    rgbs = []
    for b in bands:
        src_fp = os.path.join(f'{TMP_DIR}/{landsat_id}_{b}_4326.TIF')
        logger.debug(src_fp)
        src = rasterio.open(src_fp)
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False)
        out_meta = src.meta
        bounds = src.bounds

        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

        with rasterio.open(f"{TMP_DIR}/RGB.byte.masked_{b}.TIF", "w", **out_meta) as dest:
            dest.write(out_image)
        # rgb_raster_mask = rasterio.open(f"{TMP_DIR}/RGB.byte.masked_{b}.TIF")
        mask_imgs.append(np.squeeze(out_image, axis=0))

        channel_img = rasterio.open(f'{TMP_DIR}/{landsat_id}_{b}_4326.TIF')
        rgbs.append(normalize(np.array(channel_img.read(1))))

    if not IS_SEPARATE_BANDS:
        rgb_img = np.dstack((r for r in rgbs))
        mask_imgs = np.array(mask_imgs)
        mask = np.dstack(tuple(mask_imgs))
        return rgb_img, mask, bounds
    return rgbs, mask_imgs, bounds

def get_cropped_img(img: np.array, bottom_right_yx: tuple, top_left_yx: tuple):
    img = img[top_left_yx[0]:bottom_right_yx[0], top_left_yx[1]:bottom_right_yx[1]]
    return img

def change_axis_labels_cropped(plt, bounds, row_idx, col_idx, h, w):
    # logger.debug(f"bounds: {bounds}")
    num_y_crops, num_x_crops = h//CROPPED_TILE_SIDE_LEN, w//CROPPED_TILE_SIDE_LEN
    # logger.debug(f"num_y_crops: {num_y_crops}  num_x_crops: {num_x_crops}")
    crop_y_interval = (bounds.top - bounds.bottom)/num_y_crops
    crop_x_interval = (bounds.right - bounds.left)/num_x_crops
    left_bounds = bounds.left + crop_x_interval*col_idx
    right_bounds = bounds.left + crop_x_interval*(col_idx+1)
    bottom_bounds = bounds.bottom + crop_x_interval*row_idx
    top_bounds = bounds.bottom + crop_x_interval*(row_idx+1)
    # logger.debug(f"left_bounds:{left_bounds}, right_bounds: {right_bounds}, crop_x_interval:{crop_x_interval}")
    # logger.debug(f"bottom_bounds:{bottom_bounds}, top_bounds: {top_bounds}, crop_y_interval:{crop_y_interval}")

    xlocs, xlabels = plt.xticks()
    ylocs, ylabels = plt.yticks()
    xlocs = [i for i in range(0,CROPPED_TILE_SIDE_LEN+1, CROPPED_TILE_SIDE_LEN//(len(xlocs)-1))]
    x_interval = (right_bounds-left_bounds)/(len(xlocs)-1)
    x_labels = [f"{(left_bounds+(x_interval*i)):.03f}" for i, loc in enumerate(xlocs)]
    # logger.debug(f"x locs: {xlocs}, x_labels: {x_labels}")
    plt.xticks(ticks=xlocs, labels=x_labels)

    y_interval = (top_bounds-bottom_bounds)/(len(ylocs)-1)
    ylocs = [i for i in range(CROPPED_TILE_SIDE_LEN, -1, -1*CROPPED_TILE_SIDE_LEN//(len(ylocs)-1))]
    y_labels = [f"{(bottom_bounds+(y_interval*i)):.03f}" for i, loc in enumerate(ylocs)]
    # logger.debug(f"y locs: {ylocs}, y_labels {y_labels}")
    # print()
    plt.yticks(ticks=ylocs, labels=y_labels)

    plt.xlabel("Longitude"); plt.ylabel("Latitude")

def change_axis_labels_mul(plt, bounds, side_len, px_res=30):

    import matplotlib.ticker as mticker
    ax = plt.gca()

    xlocs, xlabels = plt.xticks()
    ylocs, ylabels = plt.yticks()

    xlocs = [i for i in range(0,side_len+1, side_len//(len(xlocs)-1))]
    x_interval = side_len/(len(xlocs)-1)
    x_labels = [float(item)*px_res/1000 for item in xlocs]
    # x_labels = [f"{(left_bounds+(x_interval*i)):.03f}" for i, loc in enumerate(xlocs)]
    # logger.debug(f"x locs: {xlocs}, x_labels: {x_labels}")
    plt.xticks(ticks=xlocs, labels=x_labels)

    y_interval = side_len/(len(ylocs)-1)
    ylocs = [i for i in range(side_len, -1, -1*side_len//(len(ylocs)-1))]
    y_labels = [float(item)*px_res/1000 for item in ylocs]
    # y_labels = [f"{(bottom_bounds+(y_interval*i)):.03f}" for i, loc in enumerate(ylocs)]
    # logger.debug(f"y locs: {ylocs}, y_labels {y_labels}")
    # print()
    plt.yticks(ticks=ylocs, labels=y_labels)

    plt.xlabel("km"); plt.ylabel("km")

def change_axis_labels(plt, bounds, h, w):
    xlocs, xlabels = plt.xticks()
    ylocs, ylabels = plt.yticks()
    xlocs = [i for i in range(0,w+1, w//(len(xlocs)-1))]
    x_interval = (bounds.right-bounds.left)/(len(xlocs)-1)
    x_labels = [f"{(bounds.left+(x_interval*i)):.02f}" for i, loc in enumerate(xlocs)]
    # logger.debug(f"x locs: {xlocs}, x_labels: {x_labels}")
    plt.xticks(ticks=xlocs, labels=x_labels)

    y_interval = (bounds.top-bounds.bottom)/(len(ylocs)-1)
    ylocs = [i for i in range(h, -1, -1*h//(len(ylocs)-1))]
    y_labels = [f"{(bounds.bottom+(y_interval*i)):.03f}" for i, loc in enumerate(ylocs)]
    # logger.debug(f"y locs: {ylocs}, y_labels {y_labels}")
    # print()
    plt.yticks(ticks=ylocs, labels=y_labels)

    plt.xlabel("Longitude"); plt.ylabel("Latitude")

# Split image into squares of size side_len by side_len
def get_smaller_crops(img, mask, side_len: int, landsat_id:str, band_name: str, 
                bounds, metadata=None) -> list:
    # logger.debug(f"get samller crops img.shape: {img.shape}")
    # logger.debug(f"bounds: {bounds}")
    if not IS_SEPARATE_BANDS:
        h, w, channels = img.shape
    else:
        h, w = img.shape
    if band_name == "B8":
        px_res = 15
        side_len = int(side_len*2)
    else:
        px_res = 30
    
    nonzero_crops = []
    r_ctr = 0
    for row_idx in range(side_len//2, h, side_len):
        c_ctr = 0
        for col_idx in range(side_len//2, w, side_len):
            top_left_yx = (row_idx, col_idx)
            bottom_right_yx = (row_idx+side_len, col_idx+side_len)

            cropped_mask = get_cropped_img(mask, bottom_right_yx, top_left_yx)
            # if np.count_nonzero(cropped_mask): # means non-zero
            if np.count_nonzero(cropped_mask) > (side_len**2)*PCT_FRAME_SAVE: # means at least 20% of frame is covered
                cropped_img = get_cropped_img(img, bottom_right_yx, top_left_yx)
                nonzero_crops.append(cropped_img)
                cropped_mask = cropped_mask/np.max(cropped_mask)
                plt.figure()
                # change_axis_labels_cropped(plt, bounds, r_ctr, c_ctr, h, w)
                if band_name == "B8":   # only do this for naming purposes to be consistent with other bands
                    row_idx = row_idx//2
                    col_idx = col_idx//2
                change_axis_labels_mul(plt, bounds, side_len, px_res=px_res)
                plt.imshow(cropped_img)
                plt.savefig(f"{OUT_DIR}/{landsat_id}/cropped_y{row_idx:02d}_x{col_idx:02d}_{band_name}_unmarked.jpg", dpi=300, bbox_inches="tight")
                is_marked = plot_metadata_cropped(plt, yxmarks, labels, dates, bottom_right_yx, top_left_yx)
                plt.savefig(f"{OUT_DIR}/{landsat_id}/cropped_y{row_idx:02d}_x{col_idx:02d}_{band_name}.jpg", dpi=300, bbox_inches="tight")
                if is_marked:
                    if not os.path.exists(f"{OUT_DIR}/{landsat_id}/marked"):
                        os.makedirs(f"{OUT_DIR}/{landsat_id}/marked")
                    plt.savefig(f"{OUT_DIR}/{landsat_id}/marked/cropped_y{row_idx:02d}_x{col_idx:02d}_{band_name}.jpg", dpi=300, bbox_inches="tight")
                plt.close()

                plt.figure()
                # change_axis_labels_cropped(plt, bounds, r_ctr, c_ctr, h, w)
                change_axis_labels_mul(plt, bounds, side_len, px_res=px_res)
                plt.imshow(cropped_mask)
                plt.savefig(f"{OUT_DIR}/{landsat_id}/cropped_y{row_idx:02d}_x{col_idx:02d}_{band_name}_mask_unmarked.jpg", dpi=300, bbox_inches="tight")
                is_marked = plot_metadata_cropped(plt, yxmarks, labels, dates, bottom_right_yx, top_left_yx)
                plt.savefig(f"{OUT_DIR}/{landsat_id}/cropped_y{row_idx:02d}_x{col_idx:02d}_{band_name}_mask.jpg", dpi=300, bbox_inches="tight")
                if is_marked:
                    if not os.path.exists(f"{OUT_DIR}/{landsat_id}/marked"):
                        os.makedirs(f"{OUT_DIR}/{landsat_id}/marked")
                    plt.savefig(f"{OUT_DIR}/{landsat_id}/marked/cropped_y{row_idx:02d}_x{col_idx:02d}_{band_name}_mask.jpg", dpi=300, bbox_inches="tight")
                plt.close()
            c_ctr += 1
        r_ctr += 1

    return nonzero_crops

# Load validation data and associated features
val_data = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))
# Iterate over all landsat directories
# for root,dirs,files in os.walk(DATA_DIR):
#     for dir in dirs:
#         if not dir.startswith("LC08"):
#             continue    # only process landsat 8 images
#         if not os.path.exists(f"{OUT_DIR}/{dir}"):
#             os.makedirs(f"{OUT_DIR}/{dir}")
#         yxmarks, labels, dates = get_tile_metadata(val_data, landsat_id=dir)
#         display_and_save_rgb_images(landsat_id=dir, yxmarks=yxmarks, labels=labels, dates=dates)

# TODO: Given the intersection, get size nxn images/rasters to have better quality images of the rivers 
# (perhaps this is something that can eventually be done with GEE API)
# NOTE: the shapefile and the tiff file should have the same CRS

# Load shapefile of rivers
logger.debug(f"Loading shapefile")
shp_fp = os.path.join(DATA_DIR, SHP_FP)
with fiona.open(shp_fp, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

# Iterate over all landsat directories
logger.debug(f"Iterating landsat directories")
for root,dirs,files in os.walk(DATA_DIR):
    for dir in dirs:
        if not dir.startswith("LC08"):
            continue    # only process landsat 8 images
        if not os.path.exists(f"{OUT_DIR}/{dir}"):
            os.makedirs(f"{OUT_DIR}/{dir}")
        yxmarks, labels, dates = get_tile_metadata(val_data, landsat_id=dir)
        bands_img, bands_mask, bounds = get_masked_raster_bands(landsat_id=dir, shapes=shapes)

        
        if not IS_SEPARATE_BANDS:
            logger.debug(bands_img.shape, bands_mask.shape)
            plt.imshow(bands_img, interpolation='none')
            plot_metadata(plt, yxmarks, labels, dates)
            plt.show()
            plt.close()

            plt.imshow(bands_mask, interpolation='none')
            plot_metadata(plt, yxmarks, labels, dates)
            plt.show()
            plt.close()
        else:
            plt.figure()
            h,w = bands_img[0].shape
            plt.imshow(bands_img[0], interpolation='none')
            plot_metadata(plt, yxmarks, labels, dates)
            change_axis_labels(plt, bounds, h, w)
            logger.debug(f"Saving to: {OUT_DIR}/{dir}/_{dir}_{BANDS[0]}.jpg")
            plt.savefig(f"{OUT_DIR}/{dir}/_{dir}_{BANDS[0]}.jpg", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            h,w = bands_mask[0].shape
            plt.imshow(bands_mask[0], interpolation='none')
            plot_metadata(plt, yxmarks, labels, dates)
            change_axis_labels(plt, bounds, h, w)
            logger.debug(f"Saving to:{OUT_DIR}/{dir}/_{dir}_{BANDS[0]}_mask.jpg")
            plt.savefig(f"{OUT_DIR}/{dir}/_{dir}_{BANDS[0]}_mask.jpg", dpi=300, bbox_inches="tight")
            plt.close()
        

        if not IS_SEPARATE_BANDS:
            get_smaller_crops(
                bands_img,
                bands_mask,
                side_len=CROPPED_TILE_SIDE_LEN,
                landsat_id=dir,
                band_name=BANDS_ID,
                bounds=bounds,
                metadata=(yxmarks, labels, dates),
            )
        else:
            for i in range(len(bands_img)):
                get_smaller_crops(
                    bands_img[i],
                    bands_mask[i],
                    side_len=CROPPED_TILE_SIDE_LEN,
                    landsat_id=dir,
                    band_name=BANDS[i],
                    bounds=bounds,
                    metadata=(yxmarks, labels, dates),
                )