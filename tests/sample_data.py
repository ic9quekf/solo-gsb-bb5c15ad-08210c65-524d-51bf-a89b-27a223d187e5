"""构造合成 DICOM ZIP 的测试夹具：CT + RTSTRUCT（深层引用）+ 非法文件。"""
from __future__ import annotations

import io
import zipfile

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTStructureSetStorage,
)


def _write_bytes(ds: FileDataset) -> bytes:
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def _ct_like(sop_uid: str, series_uid: str, study_uid: str, patient_id: str,
             patient_name: str, frame_uid: str, pixels: bytes | None = None) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = "19800115"
    ds.PatientSex = "M"
    ds.StudyDate = "20240301"
    ds.SeriesDate = "20240301"
    ds.ContentDate = "20240301"
    ds.StudyTime = "101500"
    ds.AccessionNumber = "ACC001"
    ds.StudyID = "STUDY1"
    ds.StudyDescription = "Head CT"
    ds.SeriesDescription = "Axial 5mm"
    ds.InstitutionName = "Secret Hospital"
    ds.ReferringPhysicianName = "Dr^House"
    ds.OperatorsName = "Op^One"
    ds.StationName = "CT-STATION-1"
    ds.DeviceSerialNumber = "SN-12345"
    ds.PatientAddress = "1 Secret Street"
    ds.ImageComments = "free text PHI"
    ds.Manufacturer = "Vendor"
    ds.Modality = "CT"
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if pixels is None:
        pixels = bytes(range(32))
    ds.PixelData = pixels
    # 私有标签
    block = ds.private_block(0x0009, "PHI CREATOR", create=True)
    block.add_new(0x10, "LO", "Private PHI value")
    return _write_bytes(ds)


def _rtstruct_refs(
    sop_uid: str,
    series_uid: str,
    study_uid: str,
    patient_id: str,
    patient_name: str,
    *,
    frame_uid: str | None = None,
    contour_series_uid: str | None = None,
    contour_refs: tuple[str, ...] = (),
    direct_refs: tuple[str, ...] = (),
) -> bytes:
    """构造 RTSTRUCT：深层 ContourImageSequence 引用 + 顶层 ReferencedSOPSequence。"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = "19800115"
    ds.StudyDate = "20240302"
    ds.AccessionNumber = "ACC001"
    ds.InstitutionName = "Secret Hospital"
    ds.StructureSetLabel = "RTSTRUCT"
    ds.Modality = "RTSTRUCT"

    # RT ReferencedFrameOfReferenceSequence（深层嵌套引用，任意层级）
    if contour_refs:
        rfors = Dataset()
        if frame_uid is not None:
            rfors.FrameOfReferenceUID = frame_uid
        rtrs = Dataset()
        rtrs.SeriesInstanceUID = contour_series_uid or series_uid
        rtrs.ContourImageSequence = []
        for ref in contour_refs:
            cis = Dataset()
            cis.ReferencedSOPClassUID = CTImageStorage
            cis.ReferencedSOPInstanceUID = ref
            rtrs.ContourImageSequence.append(cis)
        rfors.RTReferencedStudySequence = [rtrs]
        ds.ReferencedFrameOfReferenceSequence = [rfors]

    if direct_refs:
        ds.ReferencedSOPSequence = []
        for ref in direct_refs:
            item = Dataset()
            item.ReferencedSOPClassUID = CTImageStorage
            item.ReferencedSOPInstanceUID = ref
            ds.ReferencedSOPSequence.append(item)
    return _write_bytes(ds)


def _rtstruct_like(sop_uid: str, series_uid: str, study_uid: str, patient_id: str,
                   patient_name: str, frame_uid: str, ref_sop: str,
                   dangling_sop: str) -> bytes:
    return _rtstruct_refs(
        sop_uid, series_uid, study_uid, patient_id, patient_name,
        frame_uid=frame_uid, contour_series_uid=series_uid,
        contour_refs=(ref_sop,), direct_refs=(dangling_sop,),
    )


def build_sample_zip(path: str, seed: int = 1) -> dict:
    def uid(i: int) -> str:
        return f"1.2.3.4.5.6.7.8.9.{seed}.{i}"

    study_uid = uid(1)
    series_uid = uid(2)
    frame_uid = uid(3)
    sops = [uid(4), uid(5)]
    rt_sop = uid(6)
    dangling = uid(99)

    data = {
        "patient_1/CT0001.dcm": _ct_like(sops[0], series_uid, study_uid, "P-001", "Zhang^San", frame_uid),
        "patient_1/CT0002.dcm": _ct_like(sops[1], series_uid, study_uid, "P-001", "Zhang^San", frame_uid),
        "patient_1/rt/RS0001.dcm": _rtstruct_like(rt_sop, uid(7), study_uid,
                                                  "P-001", "Zhang^San", frame_uid,
                                                  sops[0], dangling),
        "patient_2/CT0001.dcm": _ct_like(uid(10), uid(11), uid(12),
                                         "P-002", "Li^Si", uid(13)),
        "notes/not-a-dicom.txt": b"hello, this is not dicom",
        "broken/bad.dcm": b"\x00\x01\x02not dicom at all\xff\xfe",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in data.items():
            zf.writestr(name, blob)
    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "frame_uid": frame_uid,
        "sops": sops,
        "rt_sop": rt_sop,
        "dangling": dangling,
    }


def _ct_with_refs(sop_uid: str, series_uid: str, study_uid: str, patient_id: str,
                  patient_name: str, frame_uid: str,
                  direct_refs: tuple[str, ...] = ()) -> bytes:
    """与 _ct_like 相同的 CT，但顶层带 ReferencedSOPSequence 引用。"""
    blob = _ct_like(sop_uid, series_uid, study_uid, patient_id, patient_name, frame_uid)
    # 重新读出并追加引用后写回，避免重复维护 CT 字段
    ds = pydicom.dcmread(io.BytesIO(blob))
    if direct_refs:
        ds.ReferencedSOPSequence = []
        for ref in direct_refs:
            item = Dataset()
            item.ReferencedSOPClassUID = CTImageStorage
            item.ReferencedSOPInstanceUID = ref
            ds.ReferencedSOPSequence.append(item)
    return _write_bytes(ds)


def build_multistudy_zip(path: str, *, seed: int = 100) -> dict:
    """三个检查（其中两个跨患者）+ 多跳跨检查引用的合成 ZIP。

    - 检查 A（P-001）：CT a1, a2 + RTSTRUCT ra
        ra 深层 ContourImageSequence -> a2
    - 检查 B（P-001）：CT b1
        b1 顶层 ReferencedSOPSequence -> ra（跨检查：B -> A）
    - 检查 C（P-002）：RTSTRUCT rc
        rc 深层 ContourImageSequence -> b1（跨检查：C -> B，多跳补全 ra/a2）

    以检查 B/C 为起点都应把引用链闭包补全；以检查 A 为起点则只有 A 自身。
    """
    def uid(i: int) -> str:
        return f"1.2.3.4.5.6.7.8.9.{seed}.{i}"

    study_a, series_a, frame_a = uid(1), uid(2), uid(3)
    a1, a2 = uid(4), uid(5)
    ra, ra_series = uid(6), uid(7)

    study_b, series_b = uid(20), uid(21)
    b1 = uid(22)

    study_c, series_c = uid(30), uid(31)
    rc = uid(32)

    data = {
        "study_a/CT0001.dcm": _ct_like(a1, series_a, study_a, "P-001", "Zhang^San", frame_a),
        "study_a/CT0002.dcm": _ct_like(a2, series_a, study_a, "P-001", "Zhang^San", frame_a),
        "study_a/rt/RS.dcm": _rtstruct_refs(
            ra, ra_series, study_a, "P-001", "Zhang^San",
            frame_uid=frame_a, contour_series_uid=series_a,
            contour_refs=(a2,),
        ),
        # b1 顶层 ReferencedSOPSequence -> ra（跨检查 B -> A）
        "study_b/CT0001.dcm": _ct_with_refs(
            b1, series_b, study_b, "P-001", "Zhang^San", uid(23),
            direct_refs=(ra,),
        ),
        # rc 深层 ContourImageSequence -> b1（跨检查 C -> B，多跳补全）
        "study_c/RS.dcm": _rtstruct_refs(
            rc, series_c, study_c, "P-002", "Li^Si",
            contour_refs=(b1,),
        ),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in data.items():
            zf.writestr(name, blob)
    return {
        "studies": {"A": study_a, "B": study_b, "C": study_c},
        "series": {"A": series_a, "B": series_b, "C": series_c, "ra": ra_series},
        "sops": {"a1": a1, "a2": a2, "ra": ra, "b1": b1, "rc": rc},
    }


def build_duplicate_sop_zip(path: str, *, seed: int = 200) -> dict:
    """两个文件共享同一 SOPInstanceUID（重复实例），被同检查另一个实例引用。"""
    def uid(i: int) -> str:
        return f"1.2.3.4.5.6.7.8.9.{seed}.{i}"

    study_uid, series_uid, frame_uid = uid(1), uid(2), uid(3)
    dup_sop, ref_sop = uid(4), uid(5)

    data = {
        "dup/CT0001.dcm": _ct_like(dup_sop, series_uid, study_uid, "P-001", "Zhang^San", frame_uid),
        "dup/CT0002.dcm": _ct_like(dup_sop, series_uid, study_uid, "P-001", "Zhang^San", frame_uid),
        "dup/RS.dcm": _rtstruct_refs(
            ref_sop, uid(7), study_uid, "P-001", "Zhang^San",
            frame_uid=frame_uid, contour_series_uid=series_uid,
            contour_refs=(dup_sop,),
        ),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in data.items():
            zf.writestr(name, blob)
    return {
        "study_uid": study_uid,
        "sops": {"dup": dup_sop, "ref": ref_sop},
    }
