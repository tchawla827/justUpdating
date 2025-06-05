import os
import pickle
import shutil
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import insightface
import numpy as np
from pathlib import Path
from tqdm import tqdm

import modules.globals
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.typing import Frame
from modules.utilities import (
    get_temp_directory_path,
    create_temp,
    extract_frames,
    clean_temp,
    get_temp_frame_paths,
)


FACE_ANALYSER = None
CACHE_FILE = "embeddings.pkl"


def _get_cache_path(target_path: str) -> Path:
    """Return the persistent cache path for a given target video."""
    target_dir = os.path.dirname(target_path)
    target_name = Path(target_path).stem
    cache_dir = Path(target_dir) / ".dlc_cache"
    return cache_dir / f"{target_name}_{CACHE_FILE}"


def get_face_analyser() -> Any:
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=modules.globals.execution_providers
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    """
    Return the leftmost detected face, or None if no faces.
    """
    faces = get_face_analyser().get(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> List[Any]:
    try:
        return get_face_analyser().get(frame)
    except Exception:
        return []


def has_valid_map() -> bool:
    for m in modules.globals.source_target_map:
        if "source" in m and "target" in m:
            return True
    return False


def default_source_face() -> Any:
    for m in modules.globals.source_target_map:
        if "source" in m:
            return m["source"]["face"]
    return None


def simplify_maps() -> None:
    centroids, faces = [], []
    for m in modules.globals.source_target_map:
        if "source" in m and "target" in m:
            centroids.append(m["target"]["face"].normed_embedding)
            faces.append(m["source"]["face"])
    modules.globals.simple_map = {"source_faces": faces, "target_embeddings": centroids}


def add_blank_map() -> None:
    try:
        max_id = -1
        if modules.globals.source_target_map:
            max_id = max(m["id"] for m in modules.globals.source_target_map)
        modules.globals.source_target_map.append({"id": max_id + 1})
    except Exception:
        pass


def get_unique_faces_from_target_image() -> None:
    try:
        modules.globals.source_target_map = []
        img = cv2.imread(modules.globals.target_path)
        faces = get_many_faces(img)
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            modules.globals.source_target_map.append(
                {"id": i, "target": {"cv2": crop, "face": face}}
            )
    except Exception:
        modules.globals.source_target_map = []


# ───────────────────────────────────────────────────────────────
# Serialization helpers for caching
# ───────────────────────────────────────────────────────────────
def _face_to_dict(face: Any) -> Dict[str, Any]:
    return {
        "bbox": face.bbox.tolist(),
        "kps": face.kps.tolist() if getattr(face, "kps", None) is not None else None,
        "landmark_2d_106": face.landmark_2d_106.tolist()
        if getattr(face, "landmark_2d_106", None) is not None
        else None,
        "normed_embedding": face.normed_embedding.tolist(),
        "det_score": float(face.det_score),
    }


def _dict_to_face(data: Dict[str, Any]) -> Any:
    obj = SimpleNamespace()
    obj.bbox = np.asarray(data["bbox"], dtype=np.float32)
    obj.kps = (
        np.asarray(data["kps"], dtype=np.float32) if data["kps"] is not None else None
    )
    obj.landmark_2d_106 = (
        np.asarray(data["landmark_2d_106"], dtype=np.float32)
        if data["landmark_2d_106"] is not None
        else None
    )
    obj.normed_embedding = np.asarray(data["normed_embedding"], dtype=np.float32)
    obj.det_score = data["det_score"]
    return obj


def _serialise_map(src_map: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def conv_frame(f):
        return {
            "frame": f["frame"],
            "location": f["location"],
            "faces": [_face_to_dict(fc) for fc in f["faces"]],
        }

    out = []
    for m in src_map:
        nm = {"id": m["id"]}
        if "source" in m:
            nm["source"] = {"face": _face_to_dict(m["source"]["face"])}
        if "target" in m:
            nm["target"] = {"face": _face_to_dict(m["target"]["face"])}
        if "target_faces_in_frame" in m:
            nm["target_faces_in_frame"] = [conv_frame(fr) for fr in m["target_faces_in_frame"]]
        out.append(nm)
    return out


def _deserialise_map(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def conv_frame(f):
        return {
            "frame": f["frame"],
            "location": f["location"],
            "faces": [_dict_to_face(fc) for fc in f["faces"]],
        }

    out = []
    for m in raw:
        nm = {"id": m["id"]}
        if "source" in m:
            nm["source"] = {"face": _dict_to_face(m["source"]["face"])}
        if "target" in m:
            nm["target"] = {"face": _dict_to_face(m["target"]["face"])}
        if "target_faces_in_frame" in m:
            nm["target_faces_in_frame"] = [conv_frame(fr) for fr in m["target_faces_in_frame"]]
        out.append(nm)
    return out


def get_unique_faces_from_target_video() -> None:
    cache_path = _get_cache_path(modules.globals.target_path)
    cache_dir = cache_path.parent

    # 1) Attempt to reuse cache
    if cache_path.exists():
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        extract_frames(modules.globals.target_path)
        with open(cache_path, "rb") as f:
            modules.globals.source_target_map = _deserialise_map(pickle.load(f))
        default_target_face()      # ← rebuilds each map['target']['cv2'] thumbnail
        print("[ANALYSER] Re-used cached face-embeddings")
        simplify_maps()
        return

    # 2) Fresh processing
    modules.globals.source_target_map = []
    frame_face_embeds, all_embeds = [], []

    print("Creating temp resources…")
    clean_temp(modules.globals.target_path)
    create_temp(modules.globals.target_path)

    print("Extracting frames…")
    extract_frames(modules.globals.target_path)
    frame_paths = get_temp_frame_paths(modules.globals.target_path)

    for idx, fpath in enumerate(tqdm(frame_paths, desc="Extracting embeddings")):
        img = cv2.imread(fpath)
        faces = get_many_faces(img)
        for fc in faces:
            all_embeds.append(fc.normed_embedding)
        frame_face_embeds.append({"frame": idx, "faces": faces, "location": fpath})

    centroids = find_cluster_centroids(all_embeds)

    for fr in frame_face_embeds:
        for fc in fr["faces"]:
            cid, _ = find_closest_centroid(centroids, fc.normed_embedding)
            fc.target_centroid = cid

    for cid in range(len(centroids)):
        entry = {"id": cid, "target_faces_in_frame": []}
        for fr in frame_face_embeds:
            faces_here = [fc for fc in fr["faces"] if fc.target_centroid == cid]
            entry["target_faces_in_frame"].append(
                {"frame": fr["frame"], "faces": faces_here, "location": fr["location"]}
            )
        modules.globals.source_target_map.append(entry)

    default_target_face()

    # 3) Save cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                _serialise_map(modules.globals.source_target_map),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[ANALYSER] Cached embeddings → {cache_path}")
    except Exception as e:
        print(f"[ANALYSER] WARNING: could not write cache ({e})")


def default_target_face() -> None:
    for m in modules.globals.source_target_map:
        best = None
        best_frame = None
        for fr in m.get("target_faces_in_frame", []):
            if fr["faces"]:
                best = fr["faces"][0]
                best_frame = fr
                break
        for fr in m.get("target_faces_in_frame", []):
            for fa in fr["faces"]:
                if fa.det_score > best.det_score:
                    best = fa
                    best_frame = fr
        if best is not None:
            x1, y1, x2, y2 = best.bbox
            img = cv2.imread(best_frame["location"])
            m["target"] = {"cv2": img[int(y1):int(y2), int(x1):int(x2)], "face": best}


def dump_faces(centroids: Any, frame_face_embeddings: list) -> None:
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)
    for cid in range(len(centroids)):
        dir_path = os.path.join(temp_directory_path, str(cid))
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        for fr in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/{cid}"):
            img = cv2.imread(fr["location"])
            for j, face in enumerate(fr["faces"]):
                if face.target_centroid == cid:
                    x1, y1, x2, y2 = face.bbox
                    crop = img[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        cv2.imwrite(f"{dir_path}/{fr['frame']}_{j}.png", crop)
