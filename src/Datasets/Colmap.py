"""Datasets/Colmap.py: Provides a dataset class for scenes in COLMAP format."""

from functools import partial
from pathlib import Path

import numpy as np
import pycolmap

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import RadialTangentialDistortion
from Datasets.Base import BaseDataset
from Datasets.utils import compute_scaled_image_size, read_image_size, View, ImageData, transform_poses_pca, BasicPointCloud, \
    load_inverted_segmentation_mask, load_external_binary_mask, load_external_world_normal_map, load_external_albedo_map, \
    load_external_metallic_map, load_disparity, apply_image_scale_factor, \
    load_optical_flow, apply_image_scale_factor_optical_flow, estimate_near_far, resolve_external_mask_path, \
    resolve_external_normal_path, resolve_external_albedo_path, resolve_external_metallic_roughness_path
from Logging import Logger


@Framework.Configurable.configure(
    PATH='dataset/colmap/myscene',
    IMAGE_FOLDER='images',  # image directory used for RGB supervision, relative to PATH
    USE_IMAGE_SIZE_FOR_INTRINSICS=False,  # scale COLMAP intrinsics to the selected image folder's actual resolution
    TEST_STEP=0,
    APPLY_PCA=False,
    SFM_POINTS_FILTER_RATIO=1.0,  # 0.95 works well in practice
    AABB_TOLERANCE_FACTOR=0.05,  # framework default is 0.1
    ESTIMATE_NEAR_FAR_FROM_SFM_POINTS=False,  # works well with methods that rely on tight near and far bounds
    EXTERNAL_MASKS_PATH=None,  # directory of per-image binary masks (0 ignore / 255 keep), matched to RGB filenames
    EXTERNAL_NORMALS_PATH=None,  # directory of mesh world-normal maps (n*0.5+0.5 PNG), matched to RGB filenames
    EXTERNAL_ALBEDO_PATH=None,  # directory of mesh albedo maps, matched to RGB filenames
    EXTERNAL_METALLIC_ROUGHNESS_PATH=None,  # directory of metallic/roughness maps (R=metallic), matched to RGB filenames
    TURNTABLE=Framework.ConfigParameterList(
        ENABLED=False,  # fixed-camera turntable capture: rotate scene/gaussians per frame instead of moving camera
        REFERENCE_FRAME_IDX=0,  # global frame index used as the fixed physical camera pose
    ),
)
class CustomDataset(BaseDataset):
    """Dataset class for scenes in COLMAP format."""

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for training and testing."""
        # load colmap data
        reconstruction = pycolmap.Reconstruction(self.dataset_path / 'sparse' / '0')
        Logger.log_debug(reconstruction.summary())

        external_masks_root = None
        if self.EXTERNAL_MASKS_PATH not in (None, ''):
            external_masks_root = Path(self.EXTERNAL_MASKS_PATH).expanduser()
            if not external_masks_root.is_dir():
                raise Framework.DatasetError(f'invalid EXTERNAL_MASKS_PATH: "{external_masks_root}"')
        external_normals_root = None
        if self.EXTERNAL_NORMALS_PATH not in (None, ''):
            external_normals_root = Path(self.EXTERNAL_NORMALS_PATH).expanduser()
            if not external_normals_root.is_dir():
                raise Framework.DatasetError(f'invalid EXTERNAL_NORMALS_PATH: "{external_normals_root}"')
        external_albedo_root = None
        if self.EXTERNAL_ALBEDO_PATH not in (None, ''):
            external_albedo_root = Path(self.EXTERNAL_ALBEDO_PATH).expanduser()
            if not external_albedo_root.is_dir():
                raise Framework.DatasetError(f'invalid EXTERNAL_ALBEDO_PATH: "{external_albedo_root}"')
        external_mr_root = None
        if self.EXTERNAL_METALLIC_ROUGHNESS_PATH not in (None, ''):
            external_mr_root = Path(self.EXTERNAL_METALLIC_ROUGHNESS_PATH).expanduser()
            if not external_mr_root.is_dir():
                raise Framework.DatasetError(f'invalid EXTERNAL_METALLIC_ROUGHNESS_PATH: "{external_mr_root}"')
        has_sfm_masks = Path(self.dataset_path / 'sfm_masks').exists()
        has_flow = Path(self.dataset_path / 'flow').exists()
        has_disp = Path(self.dataset_path / 'monoc_depth').exists()

        # load cameras and views
        cameras: list[PerspectiveCamera] = []
        data: list[View] = []
        global_frame_idx = 0
        for camera_idx, colmap_camera in Logger.log_progress(enumerate(reconstruction.cameras.values()), desc=f'loading camera views', leave=False, total=len(cameras)):
            # sort images belonging to this camera
            images = [image for image in reconstruction.images.values() if image.camera.camera_id == colmap_camera.camera_id]
            images = sorted(images, key=lambda image: image.name)
            if not images:
                continue
            image_folder = self.dataset_path / self.IMAGE_FOLDER
            reference_image_path = image_folder / images[0].name
            # load intrinsics
            match colmap_camera.model:
                case pycolmap.CameraModelId.SIMPLE_PINHOLE:
                    focal_x = focal_y = colmap_camera.params[0]
                    center_x = colmap_camera.params[1]
                    center_y = colmap_camera.params[2]
                    distortion = None
                case pycolmap.CameraModelId.PINHOLE:
                    focal_x = colmap_camera.params[0]
                    focal_y = colmap_camera.params[1]
                    center_x = colmap_camera.params[2]
                    center_y = colmap_camera.params[3]
                    distortion = None
                case pycolmap.CameraModelId.SIMPLE_RADIAL:
                    focal_x = focal_y = colmap_camera.params[0]
                    center_x = colmap_camera.params[1]
                    center_y = colmap_camera.params[2]
                    distortion = RadialTangentialDistortion(k1=colmap_camera.params[3])
                case pycolmap.CameraModelId.RADIAL:
                    focal_x = focal_y = colmap_camera.params[0]
                    center_x = colmap_camera.params[1]
                    center_y = colmap_camera.params[2]
                    distortion = RadialTangentialDistortion(k1=colmap_camera.params[3], k2=colmap_camera.params[4])
                case pycolmap.CameraModelId.OPENCV:
                    focal_x = colmap_camera.params[0]
                    focal_y = colmap_camera.params[1]
                    center_x = colmap_camera.params[2]
                    center_y = colmap_camera.params[3]
                    distortion = RadialTangentialDistortion(
                        k1=colmap_camera.params[4],
                        k2=colmap_camera.params[5],
                        p1=colmap_camera.params[6],
                        p2=colmap_camera.params[7],
                    )
                case _:
                    raise Framework.DatasetError(f'Camera model {colmap_camera.model} from COLMAP is not yet supported.')
            # rescale intrinsics
            reference_size = read_image_size(reference_image_path) if self.USE_IMAGE_SIZE_FOR_INTRINSICS else (colmap_camera.width, colmap_camera.height)
            width, height = compute_scaled_image_size(reference_size, self.IMAGE_SCALE_FACTOR)
            scale_factor_intrinsics_x = width / colmap_camera.width
            scale_factor_intrinsics_y = height / colmap_camera.height
            focal_x *= scale_factor_intrinsics_x
            focal_y *= scale_factor_intrinsics_y
            center_x *= scale_factor_intrinsics_x
            center_y *= scale_factor_intrinsics_y
            camera = PerspectiveCamera(
                shared_settings=self._camera_settings, width=width, height=height,
                focal_x=focal_x, focal_y=focal_y, center_x=center_x, center_y=center_y,
                distortion=distortion,
            )
            cameras.append(camera)
            # create View instances
            n_views = len(images)
            last_view_idx = n_views - 1
            idx2timestamp = 1 / last_view_idx
            for frame_idx, image in enumerate(images):
                rgb_path = image_folder / image.name
                if external_masks_root is not None:
                    mask_path = resolve_external_mask_path(external_masks_root, rgb_path, image_folder)
                    if mask_path is None:
                        raise Framework.DatasetError(
                            f'no external mask found for image "{rgb_path}" in "{external_masks_root}"'
                        )
                    segmentation = ImageData(
                        mask_path,
                        n_channels=1, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_external_binary_mask,
                        resize_fn=partial(apply_image_scale_factor, mode='nearest'),
                    )
                elif has_sfm_masks:
                    segmentation = ImageData(
                        self.dataset_path / 'sfm_masks' / f'{image.name}.png',
                        n_channels=1, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_inverted_segmentation_mask,
                    )
                else:
                    segmentation = None
                world_normal = None
                if external_normals_root is not None:
                    normal_path = resolve_external_normal_path(external_normals_root, rgb_path, image_folder)
                    if normal_path is None:
                        raise Framework.DatasetError(
                            f'no external normal map found for image "{rgb_path}" in "{external_normals_root}"'
                        )
                    world_normal = ImageData(
                        normal_path,
                        n_channels=3,
                        scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_external_world_normal_map,
                    )
                mesh_albedo = None
                if external_albedo_root is not None:
                    albedo_path = resolve_external_albedo_path(external_albedo_root, rgb_path, image_folder)
                    if albedo_path is None:
                        raise Framework.DatasetError(
                            f'no external albedo map found for image "{rgb_path}" in "{external_albedo_root}"'
                        )
                    mesh_albedo = ImageData(
                        albedo_path,
                        n_channels=3,
                        scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_external_albedo_map,
                    )
                mesh_metallic = None
                if external_mr_root is not None:
                    mr_path = resolve_external_metallic_roughness_path(external_mr_root, rgb_path, image_folder)
                    if mr_path is None:
                        raise Framework.DatasetError(
                            f'no external metallic map found for image "{rgb_path}" in "{external_mr_root}"'
                        )
                    mesh_metallic = ImageData(
                        mr_path,
                        n_channels=1,
                        scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_external_metallic_map,
                    )
                data.append(View(
                    camera=camera,
                    camera_index=camera_idx,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=image.cam_from_world().inverse().matrix(),
                    timestamp=frame_idx * idx2timestamp,  # significant assumption about how images were taken
                    rgb=ImageData(rgb_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR),
                    segmentation=segmentation,
                    forward_flow=ImageData(
                        self.dataset_path / 'flow' / f'{image.name.split(".")[0]}_forward.flo',
                        n_channels=2, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_optical_flow, resize_fn=apply_image_scale_factor_optical_flow
                    ) if has_flow and frame_idx < last_view_idx else None,
                    backward_flow=ImageData(
                        self.dataset_path / 'flow' / f'{image.name.split(".")[0]}_backward.flo',
                        n_channels=2, scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=load_optical_flow, resize_fn=apply_image_scale_factor_optical_flow
                    ) if has_flow and frame_idx > 0 else None,
                    misc=ImageData(
                        self.dataset_path / 'monoc_depth' / f'{image.name}.npy',
                        n_channels=1, load_fn=load_disparity, resize_fn=partial(apply_image_scale_factor, mode='nearest')
                    ) if has_disp else None,
                    world_normal=world_normal,
                    mesh_albedo=mesh_albedo,
                    mesh_metallic=mesh_metallic,
                ))
                global_frame_idx += 1

        # load point cloud
        self.point_cloud = BasicPointCloud.from_colmap(reconstruction)

        # rotate poses to align ground with xz plane
        if self.APPLY_PCA:
            c2ws = np.stack([view.c2w_numpy for view in data])
            c2ws, transformation = transform_poses_pca(c2ws, rescale=False)
            for view, c2w in zip(data, c2ws):
                view.c2w = c2w
            self.point_cloud.transform(transformation)
            self.scene_alignment_transform = transformation.astype(np.float64, copy=True)
            for view in data:
                view.exif['scene_alignment_transform'] = self.scene_alignment_transform

        # filter point cloud outliers
        filter_ratio = 1.0 if self.SFM_POINTS_FILTER_RATIO is None else self.SFM_POINTS_FILTER_RATIO
        if filter_ratio != 1.0:
            self.point_cloud.filter_outliers(filter_ratio)

        # extract bounding box from point cloud
        self.bounding_box = self.point_cloud.get_aabb(tolerance_factor=self.AABB_TOLERANCE_FACTOR)

        # estimate near and far plane from point cloud
        if self.ESTIMATE_NEAR_FAR_FROM_SFM_POINTS:
            self._camera_settings.near_plane, self._camera_settings.far_plane = estimate_near_far(data, self.point_cloud)

        # Convert moving-camera COLMAP turntable poses into per-frame object transforms and a fixed camera.
        # Geometry stays equivalent to the COLMAP reconstruction, but renderers can use the transform to
        # keep view-dependent appearance in the original room/environment frame.
        if self.TURNTABLE.ENABLED:
            if not data:
                raise Framework.DatasetError('TURNTABLE.ENABLED=True requires at least one COLMAP view.')
            reference_idx = int(self.TURNTABLE.REFERENCE_FRAME_IDX)
            if reference_idx < 0 or reference_idx >= len(data):
                raise Framework.DatasetError(
                    f'TURNTABLE.REFERENCE_FRAME_IDX={reference_idx} outside available view range [0, {len(data) - 1}]'
                )
            fixed_c2w = data[reference_idx].c2w_numpy
            for view in data:
                original_c2w = view.c2w_numpy
                object_transform = fixed_c2w @ view.w2c_numpy
                view.exif['turntable_original_c2w'] = original_c2w
                view.exif['turntable_fixed_c2w'] = fixed_c2w
                view.exif['turntable_object_transform'] = object_transform
                view.c2w = fixed_c2w
            Logger.log_info(
                f'enabled turntable mode with fixed camera at global frame {reference_idx}; '
                'per-view object transforms stored in View.exif'
            )

        # create splits
        dataset: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        if self.TEST_STEP > 0:
            for i in range(len(data)):
                if i % self.TEST_STEP == 0:
                    dataset['test'].append(data[i])
                else:
                    dataset['train'].append(data[i])
        else:
            dataset['train'] = data

        # return the dataset
        return cameras, dataset
