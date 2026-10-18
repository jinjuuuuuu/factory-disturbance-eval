# factory_scene.py - 공장 장면 구성과 외란 적용 (Isaac Sim 5.1 python 전용)
# ---------------------------------------------------------------------------
# 수집(collect_demos.py), 평가(eval_client.py), 점검(scene_check.py)이 모두 이
# 모듈의 build_scene()으로 장면을 만든다. 예전 프로젝트는 장면 코드가 수집과 평가에
# 각각 복사돼 있었고, 조명 값이 어긋나 성공률이 에러 없이 0%가 됐다. 장면은 여기서만
# 만든다.
#
# SimulationApp을 만든 '뒤에' import 해야 한다 (pxr/omni 모듈이 그때 로드된다).
# ---------------------------------------------------------------------------
import math
import numpy as np

import omni.usd
from pxr import UsdGeom, UsdLux, UsdShade, UsdPhysics, PhysxSchema, Gf, Sdf, Usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

import factory_config as C
from disturbances import FLICKER_LOW, vib_offset

# 정책이 화면을 보는 빈도 (수집 60Hz를 stride 3으로 줄인 20Hz). 연속 진동의 시각 계산에 쓴다.
OBS_HZ = C.RECORD_HZ / C.TRAIN_STRIDE


class Scene:
    """build_scene()이 돌려주는 핸들 묶음."""


# ---------------------------------------------------------------------------
# 장면 구성
# ---------------------------------------------------------------------------

def _add_reference(stage, path, usd, pos, rot_z=0.0, scale=1.0):
    # 에셋은 자기 xform 연산(회전 순서 ZYX 등)을 갖고 온다. 그 위에 바로 위치를 쓰면
    # 'Rotation order mismatch'로 죽는다 -> 빈 부모에 위치를 주고 에셋은 자식에 참조한다.
    prim = stage.DefinePrim(path, "Xform")
    stage.DefinePrim(path + "/Asset", "Xform").GetReferences().AddReference(usd)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*pos))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, rot_z))
    if scale != 1.0:
        api.SetScale(Gf.Vec3f(scale, scale, scale))
    return prim


def _disable_lights_under(stage, root):
    """배경 에셋에 들어 있는 조명을 끈다. 조명 외란은 우리가 둔 천장등에만 걸리므로,
    배경 조명이 켜져 있으면 외란을 줘도 장면 밝기가 덜 변한다."""
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName():
            prim.SetActive(False)
            n += 1
    return n


def _make_physics_material(stage, path, mu):
    """마찰 재질. combine mode 'min' -> 접촉 마찰 = 두 재질 중 작은 값."""
    UsdShade.Material.Define(stage, path)
    prim = stage.GetPrimAtPath(path)
    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateStaticFrictionAttr().Set(float(mu))
    mat.CreateDynamicFrictionAttr().Set(float(mu))
    mat.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    px.CreateFrictionCombineModeAttr().Set("min")
    return prim


def _bind_physics_material(target_prim, mat_path, stage):
    UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(mat_path)),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _build_tray(world):
    cx, cy = C.TRAY_CENTER_XY
    ix, iy = C.TRAY_INNER
    t, h = C.TRAY_WALL_T, C.TRAY_WALL_H
    z0 = C.TABLE_Z
    color = np.array(C.TRAY_RGB)
    parts = [
        # 바닥
        ("floor", (cx, cy, z0 + C.TRAY_FLOOR_T / 2), (ix + 2 * t, iy + 2 * t, C.TRAY_FLOOR_T)),
        # 벽 4개
        ("wall_xp", (cx + ix / 2 + t / 2, cy, z0 + h / 2), (t, iy + 2 * t, h)),
        ("wall_xn", (cx - ix / 2 - t / 2, cy, z0 + h / 2), (t, iy + 2 * t, h)),
        ("wall_yp", (cx, cy + iy / 2 + t / 2, z0 + h / 2), (ix, t, h)),
        ("wall_yn", (cx, cy - iy / 2 - t / 2, z0 + h / 2), (ix, t, h)),
    ]
    for name, pos, size in parts:
        world.scene.add(FixedCuboid(
            prim_path=f"/World/Tray/{name}", name=f"tray_{name}",
            position=np.array(pos), scale=np.array(size), size=1.0, color=color))


def _build_belt(world):
    """멈춘 작업 컨베이어: 고무 벨트 + 양쪽 프레임 레일. 로봇 앞을 가로(y방향)로 지나가며
    상자는 벨트 윗면에 놓인다."""
    y0, y1 = C.BELT_Y_RANGE
    cy, L = (y0 + y1) / 2, y1 - y0
    xc, w = C.BELT_X_CENTER, C.BELT_WIDTH
    zt = C.BELT_TOP_Z
    rz, rh = zt - 0.03 + (0.03 + C.RAIL_H) / 2, 0.03 + C.RAIL_H
    parts = [
        ("belt", (xc, cy, (C.TABLE_Z + zt) / 2), (w, L, zt - C.TABLE_Z), C.BELT_RGB),
        ("rail_near", (xc - w / 2 - 0.01, cy, rz), (0.02, L, rh), C.FRAME_RGB),
        ("rail_far", (xc + w / 2 + 0.01, cy, rz), (0.02, L, rh), C.FRAME_RGB),
    ]
    for name, pos, size, rgb in parts:
        world.scene.add(FixedCuboid(prim_path=f"/World/WorkConveyor/{name}", name=f"conv_{name}",
                                    position=np.array(pos), scale=np.array(size), size=1.0,
                                    color=np.array(rgb)))


def _disable_foreign_lights(stage, keep_root="/World/Lights"):
    """우리 조명(/World/Lights) 말고는 모두 끈다. 기본 바닥면 에셋 등에 딸린 조명이
    남아 있으면 천장등을 3000 -> 50000으로 바꿔도 밝기가 215 -> 220으로만 변한다."""
    off = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(keep_root):
            continue
        if prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName():
            prim.SetActive(False)
            off.append(p)
    return off


def _add_camera(stage, xform_path, cam_path, focal, translate, rotate):
    xf = stage.DefinePrim(xform_path, "Xform")
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(C.CAM_H_APERTURE)
    cam.GetVerticalApertureAttr().Set(C.CAM_V_APERTURE)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(C.CAM_CLIP_NEAR, C.CAM_CLIP_FAR))
    UsdGeom.XformCommonAPI(xf).SetTranslate(Gf.Vec3d(*translate))
    UsdGeom.XformCommonAPI(xf).SetRotate(Gf.Vec3f(*rotate))
    return xf


def _fix_exposure():
    """자동 노출을 끈다. 켜져 있으면 렌더러가 어두워진 장면을 다시 밝혀서, 조명을 25%로
    낮춰도 이미지 평균 밝기가 221 -> 214로만 변했다(scene_check 1차). 카메라 노출이
    고정돼 있어야 조명 외란이 이미지에 그대로 나타난다."""
    import carb
    st = carb.settings.get_settings()
    st.set("/rtx/post/histogram/enabled", False)
    st.set("/rtx/post/tonemap/exposureMode", 0)     # 0 = 수동(고정) 노출
    return {k: st.get(k) for k in ("/rtx/post/histogram/enabled", "/rtx/post/tonemap/exposureMode")}


def _bbox(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return [round(v, 3) for v in r.GetMin()], [round(v, 3) for v in r.GetMax()]


def build_scene(simulation_app, record_depth=False):
    s = Scene()
    s.exposure = _fix_exposure()
    s.app = simulation_app
    s.world = world = World(stage_units_in_meters=1.0)
    s.stage = stage = omni.usd.get_context().get_stage()

    # 바닥 충돌면: 작업대에서 떨어진 상자가 끝없이 떨어지지 않게. 공장 배경이 있으면
    # 바닥 격자 그림은 가린다(충돌은 그대로).
    world.scene.add_default_ground_plane(z_position=-0.001)
    if C.FACTORY_BG:
        gp = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if gp.IsValid():
            for p in Usd.PrimRange(gp):
                if p.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(p).MakeInvisible()
        _add_reference(stage, "/World/Warehouse", C.WAREHOUSE_USD, C.WAREHOUSE_OFFSET)
        _add_reference(stage, "/World/Conveyor", C.CONVEYOR_USD, C.CONVEYOR_POS, C.CONVEYOR_ROT_Z)
        for _ in range(30):
            simulation_app.update()           # 참조 에셋 로드
        s.n_bg_lights_off = _disable_lights_under(stage, "/World/Warehouse") \
            + _disable_lights_under(stage, "/World/Conveyor")
        s.conveyor_bbox = _bbox(stage, "/World/Conveyor")
        s.warehouse_bbox = _bbox(stage, "/World/Warehouse")
    else:
        s.n_bg_lights_off = 0
        s.conveyor_bbox = s.warehouse_bbox = None

    # 작업대
    if C.TABLE_Z > 0:
        world.scene.add(FixedCuboid(
            prim_path="/World/Workbench", name="workbench",
            position=np.array([C.TABLE_CENTER_XY[0], C.TABLE_CENTER_XY[1], C.TABLE_Z / 2]),
            scale=np.array([C.TABLE_SIZE[0], C.TABLE_SIZE[1], C.TABLE_Z]), size=1.0,
            color=np.array(C.TABLE_RGB)))

    # 로봇: 작업대 상판 위 원점
    s.franka = world.scene.add(Franka(
        prim_path="/World/Franka", name="franka",
        position=np.array([0.0, 0.0, C.TABLE_Z])))

    # 마찰 재질: 상자와 손가락 패드. 둘 다 'min' -> 접촉 마찰은 상자 값이 정한다.
    s.box_mat_path = "/World/PhysicsMaterials/BoxSurface"
    s.box_mat_prim = _make_physics_material(stage, s.box_mat_path, C.BOX_FRICTION)
    finger_mat = "/World/PhysicsMaterials/FingerPad"
    _make_physics_material(stage, finger_mat, C.FINGER_FRICTION)
    for link in ("panda_leftfinger", "panda_rightfinger"):
        fp = stage.GetPrimAtPath(f"/World/Franka/{link}")
        if fp.IsValid():
            _bind_physics_material(fp, finger_mat, stage)

    # 손가락 쥐는 힘. Isaac Sim 기본 Franka 에셋은 maxForce 7.2N이라 0.4kg 상자도 옮기다
    # 놓친다(expert_debug.py로 확인). 실제 Franka Hand 사양(연속 70N)에 맞춘다.
    s.finger_drives = []
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Franka")):
        if p.GetName().startswith("panda_finger_joint"):
            d = UsdPhysics.DriveAPI.Get(p, "linear")
            if d and d.GetMaxForceAttr().Get() is not None:
                d.GetMaxForceAttr().Set(C.FINGER_MAX_FORCE)
                d.GetStiffnessAttr().Set(C.FINGER_STIFFNESS)
                s.finger_drives.append(p.GetName())

    # 상자
    s.box = world.scene.add(DynamicCuboid(
        prim_path="/World/Box", name="box",
        position=np.array([0.45, 0.15, C.BOX_Z]), size=C.BOX_SIZE,
        color=np.array(C.BOX_RGB), mass=C.BOX_MASS))
    _bind_physics_material(stage.GetPrimAtPath("/World/Box"), s.box_mat_path, stage)
    # 질량·관성·무게중심을 런타임에 바꾸기 위한 물리 뷰
    s.box_view = world.scene.add(RigidPrim(prim_paths_expr="/World/Box", name="box_view"))

    _build_tray(world)
    _build_belt(world)

    # 조명: 천장 형광등(외란 대상) + 약한 보조 돔
    s.ceiling = []
    for i, pos in enumerate(C.CEILING_LIGHTS):
        lt = UsdLux.RectLight.Define(stage, f"/World/Lights/Ceiling{i}")
        lt.CreateWidthAttr(C.CEILING_LIGHT_SIZE[0])
        lt.CreateHeightAttr(C.CEILING_LIGHT_SIZE[1])
        lt.CreateIntensityAttr(C.CEILING_INTENSITY)
        UsdGeom.XformCommonAPI(lt.GetPrim()).SetTranslate(Gf.Vec3d(*pos))
        s.ceiling.append(lt)
    s.dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    s.dome.CreateIntensityAttr(C.AMBIENT_DOME_INTENSITY)

    # 카메라
    _add_camera(stage, "/World/Franka/panda_hand/WristCam", C.WRIST_CAM_PRIM,
                C.WRIST_FOCAL, C.WRIST_TRANSLATE, C.WRIST_ROTATE)
    s.over_xform = _add_camera(stage, C.OVER_CAM_XFORM, C.OVER_CAM_PRIM,
                               C.OVER_FOCAL, C.OVER_TRANSLATE, C.OVER_ROTATE)

    for _ in range(50):
        simulation_app.update()
    s.foreign_lights_off = _disable_foreign_lights(stage)

    import omni.replicator.core as rep
    rep.create.render_product(C.WRIST_CAM_PRIM, resolution=(C.IMG_W, C.IMG_H))
    rep.create.render_product(C.OVER_CAM_PRIM, resolution=(C.IMG_W, C.IMG_H))

    from isaacsim.sensors.camera import Camera
    s.wrist_cam = Camera(prim_path=C.WRIST_CAM_PRIM, resolution=(C.IMG_W, C.IMG_H))
    s.over_cam = Camera(prim_path=C.OVER_CAM_PRIM, resolution=(C.IMG_W, C.IMG_H))

    world.reset()
    s.wrist_cam.initialize()
    s.over_cam.initialize()
    if record_depth:
        s.wrist_cam.add_distance_to_image_plane_to_frame()
        s.over_cam.add_distance_to_image_plane_to_frame()
    for _ in range(15):
        world.step(render=True)

    s.n_dof = s.franka.num_dof
    # 질량을 배율로 바꿀 때 관성도 같은 배율로 맞추기 위한 기준값 (균일 밀도 가정)
    s.default_inertia = np.array(s.box_view.get_inertias(), dtype=np.float32).copy()
    s.frame_stats = {"blank": 0, "total": 0}
    return s


# ---------------------------------------------------------------------------
# 외란 적용
# ---------------------------------------------------------------------------

def apply_episode_disturbance(s, params, draws):
    """에피소드 시작 시(world.reset + 상자 배치 이후) 호출한다.

    외란이 없어도 반드시 호출한다 - 기준값을 다시 써서 이전 에피소드 값이 남지 않게.
    """
    # 1) 마찰
    mu = float(params["friction_mu"])
    mat = UsdPhysics.MaterialAPI(s.box_mat_prim)
    mat.GetStaticFrictionAttr().Set(mu)
    mat.GetDynamicFrictionAttr().Set(mu)

    # 2) 질량 (+ 관성 같은 배율)
    k = float(params["mass_scale"])
    s.box_view.set_masses(np.array([C.BOX_MASS * k], dtype=np.float32))
    s.box_view.set_inertias(s.default_inertia * k)

    # 3) 무게중심 (상자 로컬 좌표)
    pos, ori = s.box_view.get_coms()
    pos = np.array(pos, dtype=np.float32).reshape(1, 1, 3)
    pos[0, 0, :] = np.array(draws["com_offset"], dtype=np.float32)
    s.box_view.set_coms(positions=pos, orientations=np.array(ori, dtype=np.float32).reshape(1, 1, 4))

    # 4) 고정 카메라 거치대 틀어짐 (에피소드 동안 고정). 흔들림은 이 자세를 기준으로 떨린다.
    s.cam_base_t = [C.OVER_TRANSLATE[i] + draws["cam_offset"][i] for i in range(3)]
    s.cam_base_r = [C.OVER_ROTATE[i] + draws["cam_rot_deg"] * draws["cam_rot_axis"][i] for i in range(3)]
    UsdGeom.XformCommonAPI(s.over_xform).SetTranslate(Gf.Vec3d(*s.cam_base_t))
    UsdGeom.XformCommonAPI(s.over_xform).SetRotate(Gf.Vec3f(*s.cam_base_r))

    # 5) 조명 기준 밝기 (깜빡임은 set_lights_for_step이 매 관측마다)
    set_light_scale(s, float(params["light_scale"]))


def set_light_scale(s, scale):
    # 보조 돔(창으로 들어오는 빛 역할)도 같이 줄인다. 돔만 그대로면 어두워져도 바닥 밝기가 남는다.
    for lt in s.ceiling:
        lt.GetIntensityAttr().Set(C.CEILING_INTENSITY * scale)
    s.dome.GetIntensityAttr().Set(C.AMBIENT_DOME_INTENSITY * scale)


def set_lights_for_step(s, params, draws, k):
    """정책이 관측하는 k번째 시점의 밝기. 깜빡임이 꺼져 있으면 아무것도 바꾸지 않는다."""
    if params["flicker_prob"] <= 0:
        return
    on = draws["flicker_on"][k % len(draws["flicker_on"])]
    base = float(params["light_scale"])
    set_light_scale(s, base * (FLICKER_LOW if on else 1.0))


def set_camera_shake_for_step(s, params, draws, k):
    """정책이 관측하는 k번째 시점의 고정 카메라 떨림(무작위 흔들림 + 연속 진동).
    둘 다 꺼져 있으면 아무것도 바꾸지 않는다."""
    shake_on = params.get("shake_shift", 0) > 0 or params.get("shake_rot_deg", 0) > 0
    vib_on = params.get("vib_shift", 0) > 0 or params.get("vib_rot_deg", 0) > 0
    if not shake_on and not vib_on:
        return
    t, r = list(s.cam_base_t), list(s.cam_base_r)
    if shake_on:
        dt, dr = draws["shake"][k % len(draws["shake"])]
        t = [t[i] + dt[i] for i in range(3)]
        r = [r[i] + dr[i] for i in range(3)]
    if vib_on:
        dt, dr = vib_offset(params, draws, k, OBS_HZ)
        t = [t[i] + dt[i] for i in range(3)]
        r = [r[i] + dr[i] for i in range(3)]
    UsdGeom.XformCommonAPI(s.over_xform).SetTranslate(Gf.Vec3d(*t))
    UsdGeom.XformCommonAPI(s.over_xform).SetRotate(Gf.Vec3f(*r))


def read_back(s):
    """실제로 적용된 물리값을 읽어 결과에 남긴다 (설정했다고 믿지 말고 잰다)."""
    mat = UsdPhysics.MaterialAPI(s.box_mat_prim)
    pos, _ = s.box_view.get_coms()
    return {
        "friction_mu": round(float(mat.GetStaticFrictionAttr().Get()), 4),
        "mass": round(float(np.array(s.box_view.get_masses()).reshape(-1)[0]), 4),
        "com": [round(float(v), 4) for v in np.array(pos).reshape(-1)[:3]],
        "light": round(float(s.ceiling[0].GetIntensityAttr().Get()), 1),
    }


# ---------------------------------------------------------------------------
# 공용 유틸
# ---------------------------------------------------------------------------

def grab_rgb(s, cam):
    s.frame_stats["total"] += 1
    try:
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] < 2:
            s.frame_stats["blank"] += 1
            return np.zeros((C.IMG_H, C.IMG_W, 3), dtype=np.uint8)
        img = rgba[:, :, :3]
        if img.dtype != np.uint8:
            img = (np.clip(img * 255.0, 0, 255) if float(img.max()) <= 1.0
                   else np.clip(img, 0, 255)).astype(np.uint8)
        if float(img.std()) < 1.0:         # 사실상 단색 = 렌더가 안 붙은 것
            s.frame_stats["blank"] += 1
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        s.frame_stats["blank"] += 1
        return np.zeros((C.IMG_H, C.IMG_W, 3), dtype=np.uint8)


def get_ee_position(s):
    try:
        ee, _ = s.franka.end_effector.get_world_pose()
        return np.array(ee, dtype=np.float32)
    except Exception:
        hand = s.stage.GetPrimAtPath("/World/Franka/panda_hand")
        t = UsdGeom.Xformable(hand).ComputeLocalToWorldTransform(0).ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float32)


def box_pos(s):
    p, _ = s.box.get_world_pose()
    return np.array(p, dtype=np.float32)


def reset_episode(s, box_xy):
    s.world.reset()
    s.box.set_world_pose(position=np.array([box_xy[0], box_xy[1], C.BOX_Z], dtype=float),
                         orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    s.box.set_linear_velocity(np.zeros(3))
    s.box.set_angular_velocity(np.zeros(3))


def settle(s, n=10):
    for _ in range(n):
        s.world.step(render=True)


def move_to_start(s, steps=60):
    q = np.array(C.START_POSE, dtype=float)
    if len(q) < s.n_dof:
        cur = np.array(s.franka.get_joint_positions(), dtype=float)
        cur[:len(q)] = q
        q = cur
    for _ in range(steps):
        s.franka.apply_action(ArticulationAction(joint_positions=q))
        s.world.step(render=True)


def tray_place_target():
    return np.array([C.TRAY_CENTER_XY[0], C.TRAY_CENTER_XY[1], C.PLACE_Z + C.PLACE_Z_OFFSET])


def success_conditions(s, max_box_z):
    """성공 판정의 네 조건을 따로 돌려준다 (어느 조건에서 막혔는지 기록하려고)."""
    b = box_pos(s)
    jp7 = float(np.array(s.franka.get_joint_positions())[7])
    return {
        "lifted": max_box_z > C.BOX_Z + C.SUCCESS_MIN_LIFT,
        "in_tray": C.box_in_tray_xy(float(b[0]), float(b[1])),
        "on_floor": float(b[2]) < C.BOX_IN_TRAY_MAX_Z,
        "released": jp7 > C.GRIPPER_RELEASED_THRESH,
    }
