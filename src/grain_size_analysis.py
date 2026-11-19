import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.measure import regionprops, regionprops_table
from skimage.measure import label
from pathlib import Path

def grains_from_masks(mask_tif_path,
                      image_res=None, save_dir=None,
                      properties=('label', 'area', 'area_convex',
                                  'perimeter_crofton', 'orientation',
                                  'minor_axis_length', 'major_axis_length',
                                  'centroid', 'local_centroid',
                                  'solidity', 'eccentricity')):
    """
    Extract grain properties from a labelled segmentation mask.

    Parameters
    ----------
    mask_tif_path : str
        Path to the TIFF label mask.
        Background must be 0 and each grain must have a unique integer label.

    image_res : float, optional
        Pixel resolution in mm/pixel.
        If given, major and minor axis lengths are also returned in mm.

    properties : tuple or list
        Region properties to extract.

    Returns
    -------
    props_df : pandas.DataFrame
        DataFrame containing grain properties.

    props : list of dict
        List of dictionaries containing the same results.
    """

    # Read mask
    mask = tiff.imread(mask_tif_path)
    mask, num_grains = label(mask, return_num=True)
    print(f"Found {num_grains} grains in mask.")
    # Extract properties as table
    props_table = regionprops_table(
        mask,
        properties=properties
    )

    props_df = pd.DataFrame(props_table)

    props_df = props_df.rename(columns={
        "minor_axis_length": "minor_axis_length_px",
        "major_axis_length": "major_axis_length_px",
        "area": "area_px"
    })

    if image_res is not None:
        props_df["minor_axis_length_mm"] = props_df["minor_axis_length_px"] * image_res
        props_df["major_axis_length_mm"] = props_df["major_axis_length_px"] * image_res
        props_df["area_mm2"] = props_df["area_px"] * (image_res ** 2)

    input_path = Path(mask_tif_path)
    csv_name = f"{input_path.stem}_grains_measure.csv"

    if save_dir is None:
        # Save next to the input mask
        save_path = input_path.with_name(csv_name)
    else:
        # Save inside the specified output directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / csv_name
        
    props_df.to_csv(save_path, index=False)
    print(f"Saving grain properties to: {save_path}")
    
    return props_df

def grains_from_dataset(dataset_path,
                        suffixes=("_sam.tif"),
                        save_dir=None,
                        image_res=None,
                        recursive=True):

    dataset_path = Path(dataset_path)
    pattern = "**/*" if recursive else "*"

    mask_files = [
        p for p in dataset_path.glob(pattern)
        if p.is_file() and p.name.endswith(suffixes)
    ]

    all_results = {}

    for mask_path in mask_files:
        print(f"Processing: {mask_path}")

        props_df = grains_from_masks(
            mask_tif_path=mask_path,
            image_res=image_res,
            save_dir=save_dir
        )

        all_results[str(mask_path)] = props_df # Store DataFrame for each mask
    print(f"Number of processed masks: {len(mask_files)}")

    return all_results


def gsd_percentiles_from_csv(dataset_path,
                             pattern="*_grains_measure.csv",
                             method_name=None,
                             size_column="minor_axis_length_px",
                             percentiles=(5, 10, 16, 25, 50, 75, 84, 90, 95),
                             min_minor_axis_px=None,
                             recursive=True,
                             save=True):
    """
    Search grains_measure CSV files and compute GSD percentiles.

    Parameters
    ----------
    dataset_path : str or Path
        Dataset folder.

    pattern : str
        CSV filename pattern to search.

    size_column : str
        Column used to compute GSD percentiles.
        Example: 'minor_axis_length_px', 'major_axis_length_px',
                 'minor_axis_length_mm', 'major_axis_length_mm'

    percentiles : tuple
        Percentiles to compute.

    recursive : bool
        If True, search inside subfolders.

    save : bool
        If True, save one GSD CSV next to each grain_measure CSV.

    Returns
    -------
    gsd_df : pandas.DataFrame
        Table containing GSD percentiles for all CSV files.
    """

    dataset_path = Path(dataset_path)
    search_pattern = f"**/{pattern}" if recursive else pattern

    csv_files = list(dataset_path.glob(search_pattern))
    print(f"Found {len(csv_files)} CSV files matching pattern '{pattern}'")

    results = []

    for csv_path in csv_files:
        print(f"Processing: {csv_path}")

        df = pd.read_csv(csv_path)
        if min_minor_axis_px is not None:
         df = df[df["minor_axis_length_px"] > min_minor_axis_px]

        if size_column not in df.columns:
            print(f"Skipped: column '{size_column}' not found in {csv_path.name}")
            continue

        values = df[size_column].dropna().to_numpy()

        if len(values) == 0:
            print(f"Skipped: no valid values in {csv_path.name}")
            continue

        row = {
            "file": str(csv_path),
            "n_grains": len(values),
            "size_column": size_column
        }

        for p in percentiles:
            row[f"D{p}"] = np.percentile(values, p)

        results.append(row)

    gsd_df = pd.DataFrame(results)


    if save and len(results) > 0:
        if method_name is not None:
            global_save_path = dataset_path / f"GSD_percentiles_summary_{method_name}.csv"
        else:
            global_save_path = dataset_path / "GSD_percentiles_summary.csv"

        gsd_df.to_csv(global_save_path, index=False)
        print(f"Saved summary GSD percentiles to: {global_save_path}")

    return gsd_df

# Grain-size classification (mm)
classes = [
    ("Very fine silt", 0.0039, 0.0078),
    ("Fine silt", 0.0078, 0.0156),
    ("Medium silt", 0.0156, 0.0313),
    ("Coarse silt", 0.0313, 0.0625),
    ("Very fine sand", 0.0625, 0.125),
    ("Fine sand", 0.125, 0.25),
    ("Medium sand", 0.25, 0.5),
    ("Coarse sand", 0.5, 1.0),
    ("Very coarse sand", 1.0, 2.0),
    ("Granule", 2.0, 4.0),
    ("Pebble", 4.0, 64.0),
    ("Cobble", 64.0, 256.0),
    ("Boulder", 256.0, 4096.0),
]

def plot_histogram(curves,
                   size_column="minor_axis_length_mm",
                   min_minor_axis_px=8,
                   save_path=None,
                   show=True,
                   xmax_limit=8):
    """
    Plot grain-size histograms for one or more segmentation results.

    Grain sizes are extracted from CSV files containing grain properties.
    An optional minimum minor_axis_px length filter is applied to exclude
    grains smaller than the specified pixel threshold.

    The histograms use logarithmic size bins and distinguish grain-size
    classes according to the specified sediment classification ranges.
    Grain counts are also reported for each histogram bin and size class.

    Parameters
    ----------
    curves : list of dict
        List of dictionaries specifying the CSV file path and an optional
        label for each segmentation result.

    size_column : str, optional
        Name of the CSV column containing grain sizes in millimeters.
        Default is "minor_axis_length_mm".

    min_minor_axis_px : float or None, optional
        Minimum minor-axis length in pixels. Grains with values less than
        or equal to this threshold are excluded. If None, no pixel-based
        filtering is applied. 

    save_path : str or Path, optional

    show : bool, optional
        Whether to display the histogram.

    xmax_limit : float, optional
        Upper limit of the histogram x-axis in millimeters.

    Returns
    -------
    curve_data : list of tuple
        List containing the sorted grain-size values and corresponding
        labels for each segmentation result.
    """
    curve_data = []
    all_values = []

    for curve in curves:
        csv_path = Path(curve["csv_path"])
        label = curve.get("label", csv_path.stem)

        df = pd.read_csv(csv_path)

        if min_minor_axis_px is not None:
            if "minor_axis_length_px" not in df.columns:
                raise ValueError(f"Column 'minor_axis_length_px' not found in {csv_path.name}")
            df = df[df["minor_axis_length_px"] > min_minor_axis_px]

        if size_column not in df.columns:
            raise ValueError(f"Column '{size_column}' not found in {csv_path.name}")

        values = df[size_column].dropna().to_numpy()


        if len(values) == 0:
            raise ValueError(f"No valid grain-size values after filtering in {csv_path.name}")

        values = np.sort(values)
        curve_data.append((values, label))
        all_values.append(values)

    all_values = np.concatenate(all_values)

    visible_classes = [
        (name, xmin, xmax)
        for name, xmin, xmax in classes
        if all_values.max() >= xmin and all_values.min() <= xmax
    ]

    xmin_plot = visible_classes[0][1]
    xmax_plot = min(xmax_limit, visible_classes[-1][2])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xscale("log")
    ax.set_xlim(xmin_plot, xmax_plot)

    # Ticks principaux uniquement : limites des classes + xmax_limit
    class_ticks = []

    for name, xmin, xmax in visible_classes:
        if xmin_plot <= xmin <= xmax_plot:
            class_ticks.append(xmin)
        if xmin_plot <= xmax <= xmax_plot:
            class_ticks.append(xmax)

    if xmax_plot not in class_ticks:
        class_ticks.append(xmax_plot)

    x_ticks_display = sorted(set(class_ticks))

    # Bins histogramme : 4 sous-bins entre chaque tick principal
    bins = []
    n_sub_bins = 4

    for a, b in zip(x_ticks_display[:-1], x_ticks_display[1:]):
        sub_bins = np.logspace(np.log10(a), np.log10(b), n_sub_bins + 1)
        bins.extend(sub_bins[:-1])

    bins.append(x_ticks_display[-1])
    bins = np.array(bins)

    for values, label in curve_data:

        counts, bin_edges = np.histogram(values, bins=bins)

        print("\n" + "=" * 80)
        print(f"Results for: {label}")
        print("=" * 80)

        print("\nHistogram counts per bin:")
        for i in range(len(counts)):
            print(f"{bin_edges[i]:.6g} - {bin_edges[i+1]:.6g} mm : {counts[i]}")

        print(f"\nTotal grains in displayed histogram bins: {counts.sum()}")
        print(f"Total grains in values: {len(values)}")

        print("\nClass counts:")
        total_class_count = 0

        for class_name, xmin, xmax in visible_classes:
            xmin_c = max(xmin, xmin_plot)
            xmax_c = min(xmax, xmax_plot)

            if xmin_c >= xmax_c:
                continue

            if xmax_c == xmax_plot:
                class_count = np.sum((values >= xmin_c) & (values <= xmax_c))
            else:
                class_count = np.sum((values >= xmin_c) & (values < xmax_c))

            total_class_count += class_count
            print(f"{class_name} [{xmin_c:g}, {xmax_c:g}] mm: {class_count}")

        print(f"\nTotal over displayed classes: {total_class_count}")
        print(f"Total grains: {len(values)}")

        ax.hist(
            values,
            bins=bins,
            histtype="bar",
            alpha=0.25,
            edgecolor="gray",
            linewidth=0.4,
            label=label,
            zorder=1
        )

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.10)

    # Textes des classes après l’histogramme
    for name, xmin, xmax in visible_classes:
        xmin_c = max(xmin, xmin_plot)
        xmax_c = min(xmax, xmax_plot)

        if xmin_c >= xmax_c:
            continue

        ax.text(
            np.sqrt(xmin_c * xmax_c),
            ymax * 0.50,
            name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    # Lignes verticales uniquement aux limites de classes visibles
    for t in x_ticks_display:
        ax.axvline(t, linestyle="--", linewidth=0.6, alpha=0.5, color="gray", zorder=0)

    ax.set_xticks(x_ticks_display)
    ax.set_xticklabels([f"{t:g}" for t in x_ticks_display], rotation=45, ha="right")

    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("b-axis (mm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

    ax.legend(frameon=False, loc="upper left")

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.16, top=0.95)

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return curve_data

def plot_gsd_cumulative(
    curves,
    size_column="minor_axis_length_mm",
    min_minor_axis_px=8,
    save_path=None,
    show=True
):
    """
    Parameters
    ----------
    curves : list of dict
        List of dictionaries specifying the CSV file path and an optional
        label for each segmentation result.
    size_column : str, optional
        Column containing grain minor-axis measurements in millimetres,
    min_minor_axis_px : int or None, optional
        Minimum minor-axis length in pixels.
    save_path : str or Path, optional
    show : bool, optional
        Whether to display the plot. 

    Returns
    -------
    list of tuple
        One tuple per input curve, containing:
        - values (numpy.ndarray): Sorted grain-size measurements.
        - cumulative (numpy.ndarray): Cumulative grain percentages.
        - label (str): Curve label.
    """
    curve_data = []
    all_values = []

    for curve in curves:
        csv_path = Path(curve["csv_path"])
        label = curve.get("label", csv_path.stem)

        df = pd.read_csv(csv_path)

        if min_minor_axis_px is not None:
            if "minor_axis_length_px" not in df.columns:
                raise ValueError(f"Column 'minor_axis_length_px' not found in {csv_path.name}")
            df = df[df["minor_axis_length_px"] > min_minor_axis_px]

        if size_column not in df.columns:
            raise ValueError(f"Column '{size_column}' not found in {csv_path.name}")

        values = df[size_column].dropna().to_numpy()

        values = np.sort(values)
        cumulative = np.arange(1, len(values) + 1) / len(values) * 100

        curve_data.append((values, cumulative, label))
        all_values.append(values)

    all_values = np.concatenate(all_values)

    visible_classes = [
        (name, xmin, xmax)
        for name, xmin, xmax in classes
        if all_values.max() >= xmin and all_values.min() <= xmax
    ]

    xmin_plot = visible_classes[0][1]
    xmax_plot = visible_classes[-1][2]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xscale("log")
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(0, 100)

    class_ticks = []

    for name, xmin, xmax in visible_classes:
        ax.text(
            np.sqrt(xmin * xmax),
            50,
            name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9
        )

        class_ticks.append(xmin)
        class_ticks.append(xmax)

    class_ticks = sorted(set(class_ticks))

    extra_ticks = []

    for name, xmin, xmax in visible_classes:
        if name in ["Pebble", "Cobble", "Boulder"]:
            internal_ticks = np.logspace(np.log10(xmin), np.log10(xmax), 5)
            extra_ticks.extend(internal_ticks)

    x_ticks_display = sorted(set(class_ticks + extra_ticks))

    for values, cumulative, label in curve_data:
        ax.plot(values, cumulative, linewidth=2.2, label=label, zorder=3)

    ax.set_xticks(x_ticks_display)
    ax.set_xticklabels([f"{t:g}" for t in x_ticks_display])

    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("b-axis (mm)", fontsize=11)
    ax.set_ylabel("Cumulative count (%)", fontsize=11)

    for t in class_ticks:
        ax.axvline(
            t,
            linestyle="--",
            linewidth=0.6,
            alpha=0.5,
            color="gray",
            zorder=0
        )

    ax.legend(frameon=False, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return curve_data
