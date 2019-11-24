#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM and RF analysis for 5 burned area datasets in turn."""
from wildfires.analysis.analysis import (
    CHELSA,
    GLM,
    HYDE,
    LOGGING,
    MCD64CMQ_C6,
    RF,
    VODCA,
    AvitabileThurnerAGB,
    CCI_BurnedArea_MERIS_4_1,
    CCI_BurnedArea_MODIS_5_1,
    Copernicus_SWI,
    Datasets,
    ERA5_CAPEPrecip,
    ERA5_DryDayPeriod,
    ESA_CCI_Landcover_PFT,
    FigureSaver,
    GFEDv4,
    GFEDv4s,
    GlobFluo_SIF,
    MOD15A2H_LAI_fPAR,
    corr_plot,
    cube_plotting,
    data_processing,
    log_map,
    logger,
    logging,
    mpl,
    np,
    os,
    plot_histograms,
    print_importances,
    print_vifs,
)

if __name__ == "__main__":
    # General setup.
    logging.config.dictConfig(LOGGING)

    FigureSaver.directory = os.path.expanduser(
        os.path.join("~", "tmp", "wildfire-datasets")
    )
    os.makedirs(FigureSaver.directory, exist_ok=True)
    FigureSaver.debug = True

    # TODO: Plotting setup in a more rigorous manner.
    normal_coast_linewidth = 0.5
    mpl.rcParams["font.size"] = 9.0
    verbose = True
    np.random.seed(1)
    plot_variables = False

    # Creation of new variables.
    transformations = {
        "Temp Range": lambda exog_data: (exog_data["Max Temp"] - exog_data["Min Temp"])
    }
    # Variables to be deleted after the aforementioned transformations.
    deletions = ("Min Temp",)

    # Carry out transformations, replacing old variables in the process.
    log_var_names = ["Temp Range", "Dry Day Period"]
    sqrt_var_names = [
        # "Lightning Climatology",
        "popd"
    ]

    fire_datasets = Datasets(
        (
            fire_dataset()
            for fire_dataset in (GFEDv4s, GFEDv4, CCI_BurnedArea_MODIS_5_1, MCD64CMQ_C6)
        )
    ).select_variables(
        ["CCI MODIS BA", "GFED4 BA", "GFED4s BA", "MCD64CMQ BA"]
    ) + Datasets(
        CCI_BurnedArea_MERIS_4_1()
    ).select_variables(
        "CCI MERIS BA"
    )

    for fire_dataset, target_variable in zip(
        fire_datasets, fire_datasets.pretty_variable_names
    ):
        print(f"Target variable: {target_variable}.")
        # Dataset selection.
        # selection = get_all_datasets(ignore_names=IGNORED_DATASETS)
        # selection.remove_datasets("GSMaP Dry Day Period")
        selection = Datasets(
            (
                AvitabileThurnerAGB(),
                CHELSA(),
                Copernicus_SWI(),
                ERA5_CAPEPrecip(),
                ERA5_DryDayPeriod(),
                ESA_CCI_Landcover_PFT(),
                GFEDv4(),
                GlobFluo_SIF(),
                HYDE(),
                # LIS_OTD_lightning_climatology(),
                MOD15A2H_LAI_fPAR(),
                VODCA(),
            )
        )

        selection = selection.select_variables(
            [
                "AGBtree",
                "maximum temperature",
                "minimum temperature",
                "Soil Water Index with T=1",
                "Product of CAPE and Precipitation",
                "dry_day_period",
                "ShrubAll",
                "TreeAll",
                # "pftBare",
                "pftCrop",
                "pftHerb",
                "SIF",
                "popd",
                # "Combined Flash Rate Monthly Climatology",
                "Fraction of Absorbed Photosynthetically Active Radiation",
                "Leaf Area Index",
                "Vegetation optical depth Ku-band (18.7 GHz - 19.35 GHz)",
            ]
        )

        complete_selection = selection + fire_dataset

        (
            endog_data,
            exog_data,
            master_mask,
            filled_datasets,
            masked_datasets,
            land_mask,
        ) = data_processing(
            complete_selection,
            target_variable=target_variable,
            # XXX: Change back!!!
            which="mean",
            transformations=transformations,
            deletions=deletions,
            log_var_names=log_var_names,
            sqrt_var_names=sqrt_var_names,
            use_lat_mask=False,
            use_fire_mask=True,
        )

        # Plotting land mask.
        with FigureSaver("land_mask"):
            mpl.rcParams["figure.figsize"] = (8, 5)
            fig = cube_plotting(
                land_mask.astype("int64"), title="Land Mask", cmap="Reds_r"
            )

        # Plot histograms.
        mpl.rcParams["figure.figsize"] = (20, 12)
        plot_histograms(masked_datasets)

        # Plotting masked burned area and AGB.
        cube = masked_datasets.select_variables(target_variable, inplace=False).cube
        with FigureSaver(cube.name().replace(" ", "_")):
            mpl.rcParams["figure.figsize"] = (5, 3.33)
            fig = cube_plotting(
                cube,
                cmap="brewer_RdYlBu_11_r",
                log=True,
                label="Burnt Area Fraction",
                title=None,
                extend="min",
                coastline_kwargs={"linewidth": normal_coast_linewidth},
                boundaries=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            )

        cube = masked_datasets.select_variables("AGBtree", inplace=False).cube
        with FigureSaver(cube.name().replace(" ", "_")):
            mpl.rcParams["figure.figsize"] = (4, 2.7)
            fig = cube_plotting(
                cube,
                cmap="viridis",
                log=True,
                label=r"kg m$^{-2}$",
                title="AGBtree",
                coastline_kwargs={"linewidth": normal_coast_linewidth},
            )

        # Plotting of filled/processed Mask and AGB.
        with FigureSaver("land_mask2"):
            mpl.rcParams["figure.figsize"] = (8, 5)
            fig = cube_plotting(
                filled_datasets.select_variables(
                    target_variable, strict=True, inplace=False
                ).cube.data.mask.astype("int64"),
                title="Land Mask",
                cmap="Reds_r",
            )

        cube = filled_datasets.select_variables("AGBtree", inplace=False).cube
        with FigureSaver(cube.name().replace(" ", "_") + "_interpolated"):
            mpl.rcParams["figure.figsize"] = (4, 2.7)
            cube_plotting(
                cube,
                cmap="viridis",
                log=True,
                label=r"kg m$^{-2}$",
                title="AGBtree (interpolated)",
                coastline_kwargs={"linewidth": normal_coast_linewidth},
            )

        # Variable analysis.

        print_vifs(exog_data, thres=6)

        with FigureSaver("correlation"):
            mpl.rcParams["figure.figsize"] = (5, 3)
            corr_plot(exog_data)

        if plot_variables:
            # Creating backup slides.

            # First the original datasets.
            mpl.rcParams["figure.figsize"] = (5, 3.8)

            for cube in masked_datasets.cubes:
                with FigureSaver("backup_" + cube.name().replace(" ", "_")):
                    fig = cube_plotting(
                        cube,
                        log=log_map(cube.name()),
                        coastline_kwargs={"linewidth": normal_coast_linewidth},
                    )

            # Then the datasets after they were processed.
            for cube in filled_datasets.cubes:
                with FigureSaver("backup_mod_" + cube.name().replace(" ", "_")):
                    fig = cube_plotting(
                        cube,
                        log=log_map(cube.name()),
                        coastline_kwargs={"linewidth": normal_coast_linewidth},
                    )

        # Model Fitting.
        logger.info("Starting GLM analysis.")
        glm_results = GLM(
            endog_data,
            exog_data,
            master_mask,
            normal_coast_linewidth=normal_coast_linewidth,
        )
        logger.info("Finished GLM analysis.")

        logger.info("Starting RF analysis.")
        rf_results = RF(
            endog_data,
            exog_data,
            master_mask,
            normal_coast_linewidth=normal_coast_linewidth,
        )
        logger.info("Finished RF analysis.")

        if verbose:
            print_importances(rf_results["regr"], exog_data)
