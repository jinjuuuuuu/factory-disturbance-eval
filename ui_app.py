#!/usr/bin/env python3
# ui_app.py - 외란 주입 검증 프레임워크의 웹 화면 (Gradio)
# ---------------------------------------------------------------------------
# 슬라이더로 외란을 고르고 -> (1) 카메라 화면 미리보기 (2) 선택한 모델로 평가 실행
# (3) 일괄 실험 결과(성공률 곡선) 보기.
#
# 화면은 conda `cosmosgui`(Gradio가 있는 환경)로 띄우고, 시뮬레이션과 정책은 지금처럼
# 각자 환경(Isaac Sim python / conda lerobot)의 별도 프로세스로 돈다. 그래서 화면에서
# 돌린 평가와 명령어로 돌린 평가는 완전히 같은 코드다.
#
#   ~/miniforge3/envs/cosmosgui/bin/python ui_app.py      # http://localhost:7870
# 내 PC에서 보려면:  ssh -L 7870:localhost:7870 ws
# ---------------------------------------------------------------------------
import os
import glob
import json
import time
import socket
import subprocess
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/data/jinju/factory"
ISAAC = "/data/isaacsim/python.sh"
LEROBOT_PY = os.path.expanduser("~/miniforge3/envs/lerobot/bin/python")
PORT_UI = int(os.environ.get("UI_PORT", "7870"))
WORK = os.path.join(D, "ui_runs")
os.makedirs(WORK, exist_ok=True)

import sys
sys.path.insert(0, HERE)
import disturbances as DI


def find_models():
    """학습된 체크포인트 목록. models.json이 있으면 그 순서(최고 체크포인트)를 앞에 둔다."""
    out = {}
    mj = os.path.join(D, "models.json")
    if os.path.exists(mj):
        out.update(json.load(open(mj)))
    # v1(첫 배치) 모델은 장면이 달라 지금 장면에서 평가하면 의미가 없으므로 넣지 않는다.
    for m in ("act_v2_s1", "act_v2_s2", "act_v2_s3", "dp_v2_s1", "dp_v2_s2", "dp_v2_s3", "pi0_v2_s1"):
        # 마지막(본 실험에 쓰는) 체크포인트가 먼저 오게
        for c in sorted(glob.glob(os.path.join(D, m, "checkpoints", "0*")), reverse=True):
            out.setdefault(f"{m} @ {os.path.basename(c)}", os.path.join(c, "pretrained_model"))
    return out


EVAL_ROOT = os.environ.get("EVAL_ROOT", os.path.join(D, "eval_v2"))


def model_choices():
    """학습이 끝나 체크포인트가 생긴 모델을 매번 다시 찾는다 (앱을 다시 띄우지 않아도 되게)."""
    global MODELS
    MODELS = find_models()
    return gr.update(choices=list(MODELS), value=(list(MODELS) or [None])[0])


def add_model(name, path):
    """화면에서 모델을 추가한다: 이름과 체크포인트 폴더를 models.json에 적고 목록을 새로 고친다."""
    global MODELS
    name, path = (name or "").strip(), (path or "").strip().rstrip("/")
    if not name or not path:
        return gr.update(), "이름과 체크포인트 폴더를 모두 입력하세요."
    if os.path.isfile(os.path.join(path, "pretrained_model", "config.json")):
        path = os.path.join(path, "pretrained_model")       # checkpoints/<스텝> 까지만 넣어도 된다
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return gr.update(), f"config.json이 없습니다: {path} (LeRobot 체크포인트의 pretrained_model 폴더를 넣어야 합니다)"
    ptype = json.load(open(cfg)).get("type", "?")
    mj = os.path.join(D, "models.json")
    saved = json.load(open(mj)) if os.path.exists(mj) else {}
    saved[name] = path
    with open(mj, "w") as f:
        json.dump(saved, f, indent=1, ensure_ascii=False)
    MODELS = find_models()
    return gr.update(choices=list(MODELS), value=name), f"추가했습니다: {name} ({ptype})"


MODELS = find_models()
_servers = {}            # 모델 경로 -> (Popen, port)


def _port_open(port):
    with socket.socket() as sk:
        return sk.connect_ex(("127.0.0.1", port)) == 0


def ensure_server(path):
    """선택한 모델의 정책 서버를 띄워 두고 재사용한다."""
    if path in _servers and _servers[path][0].poll() is None:
        return _servers[path][1]
    port = 5800 + len(_servers)
    while _port_open(port):
        port += 1
    env = dict(os.environ, MODEL_PATH=path, POLICY_PORT=str(port))
    log = open(os.path.join(WORK, f"server_{port}.log"), "w")
    p = subprocess.Popen([LEROBOT_PY, "policy_server.py"], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    _servers[path] = (p, port)
    return port


def dist_string(mu, light, flicker, cam_cm, cam_deg, vib_mm, vib_deg, mass, com_cm):
    return (f"friction_mu={mu},light_scale={light / 100:.3f},flicker_prob={flicker / 100:.3f},"
            f"cam_shift={cam_cm / 100:.3f},cam_rot_deg={cam_deg},"
            f"vib_shift={vib_mm / 1000:.4f},vib_rot_deg={vib_deg},"
            f"mass_scale={mass},com_shift={com_cm / 100:.4f}")


def preview(mu, light, flicker, cam_cm, cam_deg, vib_mm, vib_deg, mass, com_cm):
    out = os.path.join(WORK, f"pv_{int(time.time())}")
    dist = dist_string(mu, light, flicker, cam_cm, cam_deg, vib_mm, vib_deg, mass, com_cm)
    env = dict(os.environ, DIST=dist, PREVIEW_OUT=out)
    yield None, None, f"미리보기 렌더링 중... (Isaac Sim 시작에 30~60초)\nDIST={dist}"
    r = subprocess.run([ISAAC, "-u", "checks/preview.py"], cwd=HERE, env=env, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if "PREVIEW_DONE" in l), "실패 - 로그 확인")
    yield (os.path.join(out, "over.png") if os.path.exists(os.path.join(out, "over.png")) else None,
           os.path.join(out, "wrist.png") if os.path.exists(os.path.join(out, "wrist.png")) else None,
           f"적용된 물리값: {line.replace('PREVIEW_DONE ', '')}\nDIST={dist}")


def run_eval(model, n_ep, mu, light, flicker, cam_cm, cam_deg, vib_mm, vib_deg, mass, com_cm):
    if model not in MODELS:
        yield None, "모델을 고르세요.", None
        return
    dist = dist_string(mu, light, flicker, cam_cm, cam_deg, vib_mm, vib_deg, mass, com_cm)
    port = ensure_server(MODELS[model])
    tag = f"ev_{int(time.time())}"
    out_json = os.path.join(WORK, tag + ".json")
    gif = os.path.join(WORK, tag + ".gif")
    env = dict(os.environ, DIST=dist, EVAL_N=str(int(n_ep)), OUT_JSON=out_json, POLICY_PORT=str(port),
               MODEL_LABEL=model, RECORD_GIF=gif)
    p = subprocess.Popen([ISAAC, "-u", "eval_client.py"], cwd=HERE, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = [f"모델 {model} | 에피소드 {int(n_ep)} | DIST={dist}", "Isaac Sim 시작 중..."]
    yield None, "\n".join(log), None
    for line in p.stdout:
        if line.startswith("[eval]"):
            log.append(line.rstrip())
            yield (gif if os.path.exists(gif) else None), "\n".join(log[-40:]), None
    p.wait()
    if not os.path.exists(out_json):
        yield None, "\n".join(log + ["평가 실패 - 로그를 확인하세요."]), None
        return
    s = json.load(open(out_json))
    summary = {"성공률": f"{s['n_success']}/{s['n_episodes']} ({100 * s['success_rate']:.0f}%)",
               "실패 유형": s["failure_modes"], "검은 프레임": s["blank_camera_frames"],
               "적용값(첫 에피소드)": s["episodes"][0]["applied"]}
    yield (gif if os.path.exists(gif) else None), "\n".join(log[-40:]), summary


def load_sweeps():
    """일괄 평가 결과를 외란(조합)별 성공률 표로 모은다. 분류 규칙은 analyze.py와 같다."""
    import pandas as pd
    from analyze import family_level, FAMS
    rows = []
    for f in glob.glob(os.path.join(EVAL_ROOT, "*", "*.json")):
        model = os.path.basename(os.path.dirname(f))
        if model.startswith("ckpt_"):
            continue
        s = json.load(open(f, encoding="utf-8"))
        fam, level = family_level(s["levels"])
        if fam is None:
            continue
        for fm in (FAMS if fam == "clean" else [fam]):     # 외란 없음은 모든 곡선의 0단계
            if fm in FAMS:
                rows.append({"model": model, "disturbance": fm, "level": level,
                             "success_rate": round(100 * s["success_rate"], 1)})
    return pd.DataFrame(rows, columns=["model", "disturbance", "level", "success_rate"])


def show_curves(fam):
    df = load_sweeps()
    if df.empty:
        return None, df
    sub = df[df.disturbance == fam].sort_values(["model", "level"])
    return sub, sub


with gr.Blocks(title="공장 외란 주입 검증") as app:
    gr.Markdown("## 공장 외란 주입 검증 프레임워크\n슬라이더로 외란을 고르고, 미리보기로 장면을 확인한 뒤, 모델을 골라 평가를 실행합니다.")
    with gr.Tab("외란 설정과 평가"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    model = gr.Dropdown(list(MODELS), label="모델 (학습 끝난 것만 표시)",
                                        value=(list(MODELS) or [None])[0], scale=4)
                    b_ref = gr.Button("목록 새로고침", scale=1)
                with gr.Accordion("모델 추가 (LeRobot 체크포인트)", open=False):
                    new_name = gr.Textbox(label="이름", placeholder="예: smolvla_s1")
                    new_path = gr.Textbox(label="체크포인트 폴더",
                                          placeholder="/data/jinju/factory/<모델>/checkpoints/<스텝>/pretrained_model")
                    b_add = gr.Button("추가")
                    add_msg = gr.Markdown()
                n_ep = gr.Slider(1, 30, value=5, step=1, label="평가 에피소드 수")
                gr.Markdown("**시각 외란** (카메라 영상이 달라짐)")
                light = gr.Slider(10, 100, value=100, step=5, label="조명 밝기 % (100 = 외란 없음)")
                flicker = gr.Slider(0, 50, value=0, step=5, label="조명 깜빡임 확률 %")
                cam_cm = gr.Slider(0, 5, value=0, step=0.5, label="카메라 틀어짐: 이동 (cm)")
                cam_deg = gr.Slider(0, 5, value=0, step=0.5, label="카메라 틀어짐: 회전 (도)")
                vib_mm = gr.Slider(0, 10, value=0, step=1, label="카메라 진동: 이동 진폭 (mm)")
                vib_deg = gr.Slider(0, 1.0, value=0, step=0.1, label="카메라 진동: 회전 진폭 (도)")
                gr.Markdown("**물리 외란** (상자를 잡고 옮기는 힘이 달라짐)")
                mu = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="상자 마찰계수 (1.0 = 외란 없음)")
                mass = gr.Slider(1.0, 6.0, value=1.0, step=0.5, label="상자 질량 배율 (기준 0.2kg)")
                com = gr.Slider(0, 2.0, value=0, step=0.2, label="무게중심 쏠림 (cm) - 질량 3배와 함께 권장")
                with gr.Row():
                    b_pv = gr.Button("미리보기")
                    b_run = gr.Button("평가 실행", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    img_over = gr.Image(label="고정 카메라", height=260)
                    img_wrist = gr.Image(label="손목 카메라", height=260)
                info = gr.Textbox(label="상태", lines=3)
                gif = gr.Image(label="첫 에피소드 (고정 카메라)", height=260)
                summary = gr.JSON(label="결과 요약")
                log = gr.Textbox(label="에피소드 로그", lines=10)
        sliders = [mu, light, flicker, cam_cm, cam_deg, vib_mm, vib_deg, mass, com]
        b_pv.click(preview, sliders, [img_over, img_wrist, info])
        b_run.click(run_eval, [model, n_ep] + sliders, [gif, log, summary])
        b_ref.click(model_choices, None, model)
        b_add.click(add_model, [new_name, new_path], [model, add_msg])
    with gr.Tab("일괄 실험 결과"):
        fam = gr.Dropdown(DI.MAIN + ["all", "friction_com", "light_flicker", "camera_vib", "mass_com",
                                   "friction_mass3"], value="camera", label="외란 종류 (단일·7종 동시·현장 조합)")
        b_ld = gr.Button("불러오기")
        plot = gr.LinePlot(x="level", y="success_rate", color="model", label="외란 단계별 성공률 (%)",
                           y_lim=[0, 100], height=360)
        table = gr.Dataframe(label="표")
        b_ld.click(show_curves, fam, [plot, table])

if __name__ == "__main__":
    # 결과 이미지·GIF가 /data/jinju/factory/ui_runs 에 생기므로 허용 경로로 열어 둔다
    # (없으면 Gradio가 InvalidPathError로 화면에 '오류'만 띄운다).
    app.queue().launch(server_name="127.0.0.1", server_port=PORT_UI, allowed_paths=[WORK])
