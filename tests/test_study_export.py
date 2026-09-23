"""按匿名 StudyInstanceUID 下载检查子集：引用闭包、跨检查补入、404/409 口径。"""
from __future__ import annotations

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.crypto import derive_uid
from tests.sample_data import (
    build_duplicate_sop_zip,
    build_multistudy_zip,
    build_sample_zip,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DICOM_ANON_JOB_TTL_SECONDS", "3600")
    import importlib
    import app.config
    import app.jobs
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _upload_zip(client, path, secret="subset-secret-12345"):
    with open(path, "rb") as fh:
        r = client.post(
            "/jobs",
            files={"file": (os.path.basename(path), fh.read(), "application/zip")},
            data={"secret": secret},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "ready", body
    return body


def _output_map(job: dict) -> dict[str, dict]:
    """原始文件名 -> {output_name, study/sop 匿名 UID}。"""
    return {
        f["name"]: {
            "output_name": f["output_name"],
            "study": f["study_instance_uid"],
            "sop": f["sop_instance_uid"],
        }
        for f in job["files"] if f["output_name"]
    }


@pytest.fixture()
def multistudy_job(client, tmp_path):
    p = tmp_path / "multi.zip"
    meta = build_multistudy_zip(str(p))
    job = _upload_zip(client, str(p))
    return client, job, _output_map(job), meta


# --------------------------------------------------------------------------- #
# 成功路径：初选 + 跨检查递归补入
# --------------------------------------------------------------------------- #
def test_study_download_primary_only(multistudy_job):
    client, job, outs, meta = multistudy_job
    study_a = outs["study_a/CT0001.dcm"]["study"]
    r = client.get(f"/jobs/{job['id']}/studies/{study_a}/download")
    assert r.status_code == 200, r.text
    # 检查 A 自身 3 个实例，无补入
    assert r.headers["X-Study-Primary-Count"] == "3"
    assert r.headers["X-Study-Added-Count"] == "0"
    assert r.headers["X-Study-Total-Count"] == "3"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert names == sorted(names)
    assert set(names) == {
        outs["study_a/CT0001.dcm"]["output_name"],
        outs["study_a/CT0002.dcm"]["output_name"],
        outs["study_a/rt/RS.dcm"]["output_name"],
    }
    # 扁平 ZIP：不含目录
    assert all("/" not in n for n in names)
    assert "attachment" in r.headers["content-disposition"]
    assert study_a in r.headers["content-disposition"]


def test_study_download_cross_study_single_hop(multistudy_job):
    # 检查 B 的 b1 引用检查 A 的 ra：补入 ra；ra 又引用 a2：再补入 a2
    client, job, outs, meta = multistudy_job
    study_b = outs["study_b/CT0001.dcm"]["study"]
    r = client.get(f"/jobs/{job['id']}/studies/{study_b}/download")
    assert r.status_code == 200, r.text
    assert r.headers["X-Study-Primary-Count"] == "1"
    assert r.headers["X-Study-Added-Count"] == "2"
    assert r.headers["X-Study-Total-Count"] == "3"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
    assert names == {
        outs["study_b/CT0001.dcm"]["output_name"],
        outs["study_a/rt/RS.dcm"]["output_name"],
        outs["study_a/CT0002.dcm"]["output_name"],
    }


def test_study_download_multi_hop_closure_across_patients(multistudy_job):
    # 检查 C（P-002）的 rc 深层引用检查 B 的 b1，b1 -> ra -> a2，闭包补全 3 个
    client, job, outs, meta = multistudy_job
    study_c = outs["study_c/RS.dcm"]["study"]
    r = client.get(f"/jobs/{job['id']}/studies/{study_c}/download")
    assert r.status_code == 200, r.text
    assert r.headers["X-Study-Primary-Count"] == "1"
    assert r.headers["X-Study-Added-Count"] == "3"
    assert r.headers["X-Study-Total-Count"] == "4"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
    assert names == {
        outs[k]["output_name"]
        for k in ("study_c/RS.dcm", "study_b/CT0001.dcm",
                  "study_a/rt/RS.dcm", "study_a/CT0002.dcm")
    }

    # ZIP 内文件为匿名 DICOM，跨检查实例确实来自不同匿名检查
    import pydicom
    studies_in_zip = set()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in names:
            ds = pydicom.dcmread(io.BytesIO(zf.read(name)), stop_before_pixels=True)
            studies_in_zip.add(str(ds.StudyInstanceUID))
    study_a = outs["study_a/CT0001.dcm"]["study"]
    study_b = outs["study_b/CT0001.dcm"]["study"]
    assert studies_in_zip == {study_c, study_b, study_a}


def test_study_download_temp_zip_cleaned(multistudy_job):
    client, job, outs, meta = multistudy_job
    study_b = outs["study_b/CT0001.dcm"]["study"]
    r = client.get(f"/jobs/{job['id']}/studies/{study_b}/download")
    assert r.status_code == 200
    # 后台任务应删除临时子集 ZIP（仅复用任务 output_dir，不留临时文件）
    job_obj = _registry().get(job["id"])
    leftovers = [
        n for n in os.listdir(os.path.dirname(job_obj.zip_path))
        if n.startswith("study-") and n.endswith(".zip")
    ]
    assert leftovers == []


def test_full_job_download_unchanged(multistudy_job):
    # 原有全量下载不受影响：5 个实例全部输出
    client, job, outs, meta = multistudy_job
    r = client.get(f"/jobs/{job['id']}/download")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert len(zf.namelist()) == 5


# --------------------------------------------------------------------------- #
# 404 / 409
# --------------------------------------------------------------------------- #
def test_study_download_job_not_found(client):
    r = client.get("/jobs/nope/studies/2.25.1/download")
    assert r.status_code == 404


def test_study_download_study_not_found_404(multistudy_job):
    client, job, outs, meta = multistudy_job
    r = client.get(f"/jobs/{job['id']}/studies/2.25.unknown/download")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "STUDY_NOT_FOUND"
    assert detail["study_instance_uid"] == "2.25.unknown"


def test_study_download_failed_job_409(client):
    r = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": "subset-secret-12345"},
    )
    job = r.json()
    resp = client.get(f"/jobs/{job['id']}/studies/2.25.1/download")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NOT_COMPLETED"


def test_study_download_missing_reference_409(client, tmp_path):
    # 样本中 RS 引用包外 dangling SOP：闭包不完整 -> 409，给出匿名 UID
    p = tmp_path / "sample.zip"
    meta = build_sample_zip(str(p))
    job = _upload_zip(client, str(p))
    outs = _output_map(job)
    study_uid = outs["patient_1/CT0001.dcm"]["study"]

    r = client.get(f"/jobs/{job['id']}/studies/{study_uid}/download")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "INCOMPLETE_REFERENCE_CLOSURE"
    assert detail["study_instance_uid"] == study_uid
    # 初选 3 个（两 CT + RT），无成功补入
    assert detail["primary_file_count"] == 3
    assert detail["added_file_count"] == 0
    conflicts = detail["conflicts"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["code"] == "MISSING_REFERENCED_INSTANCE"
    assert conflict["output_names"] == []
    # 返回的必须是匿名 UID（引用在匿名输出中已被一致重写）
    assert conflict["referenced_sop_instance_uid"] == derive_uid(
        "subset-secret-12345", meta["dangling"]
    )
    assert conflict["referenced_sop_instance_uid"] != meta["dangling"]
    # 引用来自 RT 的扁平输出名
    assert conflict["referenced_by"] == [outs["patient_1/rt/RS0001.dcm"]["output_name"]]
    # 409 不输出 ZIP
    assert r.headers["content-type"] != "application/zip"


def test_study_download_ambiguous_reference_409(client, tmp_path):
    # 同一 SOP UID 对应两个输出文件（重复实例），且被同检查实例引用
    p = tmp_path / "dup.zip"
    meta = build_duplicate_sop_zip(str(p))
    job = _upload_zip(client, str(p))
    outs = _output_map(job)
    study_uid = outs["dup/RS.dcm"]["study"]

    r = client.get(f"/jobs/{job['id']}/studies/{study_uid}/download")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "INCOMPLETE_REFERENCE_CLOSURE"
    assert detail["primary_file_count"] == 3
    codes = {c["code"] for c in detail["conflicts"]}
    assert codes == {"AMBIGUOUS_REFERENCED_INSTANCE"}
    conflict = next(
        c for c in detail["conflicts"]
        if c["code"] == "AMBIGUOUS_REFERENCED_INSTANCE"
    )
    dup_new = outs["dup/CT0001.dcm"]["sop"]
    assert conflict["referenced_sop_instance_uid"] == dup_new
    # 一个匿名 SOP UID 对应两个扁平输出
    assert sorted(conflict["output_names"]) == sorted(
        [outs["dup/CT0001.dcm"]["output_name"],
         outs["dup/CT0002.dcm"]["output_name"]]
    )
    assert conflict["referenced_by"] == [outs["dup/RS.dcm"]["output_name"]]


def test_study_download_other_study_not_pulled_without_reference(
    multistudy_job, client, tmp_path
):
    # 默认样本：患者 2 的检查与患者 1 的检查互不引用，下载检查 1 不应带入检查 2
    p = tmp_path / "sample.zip"
    from tests.sample_data import build_sample_zip
    build_sample_zip(str(p))
    job = _upload_zip(client, str(p))
    outs = _output_map(job)
    study1 = outs["patient_1/CT0001.dcm"]["study"]

    # 患者 1 的 RS 有悬空引用 -> 409（引用不完整），这里换一个无悬空的起点：
    # 患者 2 的 CT 不引用任何实例，应只下载自身
    study2 = outs["patient_2/CT0001.dcm"]["study"]
    r = client.get(f"/jobs/{job['id']}/studies/{study2}/download")
    assert r.status_code == 200, r.text
    assert r.headers["X-Study-Primary-Count"] == "1"
    assert r.headers["X-Study-Added-Count"] == "0"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert set(zf.namelist()) == {outs["patient_2/CT0001.dcm"]["output_name"]}
    assert study1 != study2


def _registry():
    from app.main import registry
    return registry
