import astra
import numpy as np
import numpy.typing as npt


def add_noise_to_sinogram(
    sinogram: npt.NDArray[np.float32],
    background_intensity: int | float,
) -> npt.NDArray[np.float32]:
    """Add Poisson noise to a sinogram.

    Notes:
        - This is a convenience wrapper for Astra Toolbox's add_noise_to_sino function.
        - A higher I_0 causes less noise, a lower I_0 causes more noise
        - Sinogram converted to attenuation form (I_0 * exp{-sino}) before added noise

    Args:
        sinogram: input sinogram as NDArray
        background_intensity: I_0/blank scan factor

    Returns:
        Noisy sinogram
    """
    return astra.functions.add_noise_to_sino(
        sinogram_in=sinogram,
        I0=background_intensity,
    )


def radon(
    img: npt.NDArray,
    views: int = 180,
    angle: float = 180,
    detector_count: int = 128,
    detector_spacing: float = 1.0,
    beam_geometry: str = "parallel",
) -> npt.NDArray[np.float32]:
    """Perform forward Radon transform.

    Args:
        img: image for forward projection
        views: number of views (i.e. height of sinogram)
        angle: maximum angle of views (degrees)
        detector_count: number of detectors (i.e. width of sinogram)
        detector_spacing: spacing between detectors
        beam_geometry: beam geometry (`parallel` or `fanflat`)

    Returns:
        Sinogram
    """
    angle = angle * np.pi / 180

    # Get image geometry and convert to Astra ID
    vol_geom = astra.create_vol_geom(img.shape)

    # Set up detector geometry
    angles = np.linspace(0, angle, views, endpoint=False)
    sino_geom = astra.create_proj_geom(
        beam_geometry,
        detector_spacing,
        detector_count,
        angles,
    )

    try:
        radon_id = astra.create_projector(
            proj_type="strip",
            proj_geom=sino_geom,
            vol_geom=vol_geom,
        )
        sinogram_id, sinogram = astra.create_sino(data=img, proj_id=radon_id)

    except Exception as e:
        raise e

    finally:
        # Clean up
        astra.data2d.delete(radon_id)
        astra.data2d.delete(sinogram_id)

    return sinogram


def iradon(
    sinogram: npt.NDArray,
    img_dims: tuple[int, int],
    recon_type: str = "FBP",
    views: int = 180,
    angle: float = 180,
    detector_count: int = 128,
    detector_spacing: float = 1.0,
    beam_geometry: str = "parallel",
) -> npt.NDArray[np.float32]:
    """Perform inverse Radon transform.

    Args:
        img: image for forward projection
        views: number of views (i.e. height of sinogram)
        angle: maximum angle of views (degrees)
        detector_count: number of detectors (i.e. width of sinogram)
        detector_spacing: spacing between detectors
        beam_geometry: beam geometry (`parallel` or `fanflat`)

    Returns:
        Reconstructed image
    """
    angle = angle * np.pi / 180

    # Get image geometry and convert to Astra ID
    volume_geom = astra.create_vol_geom(img_dims)

    # Set up detector geometry
    angles = np.linspace(0, angle, views, endpoint=False)
    sinogram_geometry = astra.create_proj_geom(
        beam_geometry,
        detector_spacing,
        detector_count,
        angles,
    )

    try:
        # Set up data IDs
        recon_img_id = astra.data2d.create(datatype="-vol", geometry=volume_geom)
        sinogram_id = astra.data2d.create(
            "-sino",
            geometry=sinogram_geometry,
            data=sinogram,
        )
        radon_id = astra.create_projector(
            proj_type="strip",
            proj_geom=sinogram_geometry,
            vol_geom=volume_geom,
        )

        # Configure and run back-projection
        cfg = {
            "ReconstructionDataId": recon_img_id,
            "ProjectionDataId": sinogram_id,
            "ProjectorId": radon_id,
            "type": recon_type,
        }
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Save data
        recon_img = astra.data2d.get(recon_img_id)

    except Exception as e:
        raise e

    finally:
        # Clean up
        astra.data2d.delete(recon_img_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(radon_id)
        astra.algorithm.delete(alg_id)

    return recon_img
