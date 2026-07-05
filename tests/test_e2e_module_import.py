"""Verify all E2E.module public APIs are importable."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_module_direct_import():
    """All symbols importable from E2E.module directly."""
    from E2E.module import (
        InceptionI3d, Unit3D, MaxPool3dSamePadding, InceptionModule,
        load_single_frame, load_frames_from_dir, load_frames_from_video,
        find_videos_from_dir, find_frames_from_dir,
        compute_optical_flow,
        build_windows, load_i3d_model, extract_features_for_video,
        run_tridet_inference, save_results_txt, THUMOS14_LABEL_NAMES,
    )
    assert InceptionI3d is not None
    assert load_frames_from_video is not None
    assert compute_optical_flow is not None
    assert build_windows is not None
    assert run_tridet_inference is not None
    print('E2E.module direct import: 19 symbols OK')


def test_shim_import():
    """Legacy `from E2E import ...` still works via shim."""
    from E2E import (
        InceptionI3d,
        load_frames_from_video,
        compute_optical_flow,
        build_windows,
        run_tridet_inference,
    )
    assert InceptionI3d is not None
    assert load_frames_from_video is not None
    print('E2E shim import: 5 symbols OK')


def test_submodule_import():
    """Individual submodules are importable."""
    from E2E.module import i3d, loader, flow, features, inference, visualizer
    assert i3d.InceptionI3d is not None
    assert loader.load_frames_from_video is not None
    assert flow.compute_optical_flow is not None
    assert features.build_windows is not None
    assert inference.run_tridet_inference is not None
    assert visualizer.create_annotated_video is not None
    print('E2E.module submodules: 6/6 OK')


def test_visualizer_submodule():
    """Visualizer is importable as submodule (not in __all__ but accessible)."""
    from E2E.module.visualizer import (
        create_annotated_video,
        predictions_to_action_list,
    )
    assert create_annotated_video is not None
    assert predictions_to_action_list is not None
    print('E2E.module.visualizer: 2 symbols OK')


if __name__ == '__main__':
    test_module_direct_import()
    test_shim_import()
    test_submodule_import()
    test_visualizer_submodule()
    print('\n=== All E2E import tests passed! ===')
