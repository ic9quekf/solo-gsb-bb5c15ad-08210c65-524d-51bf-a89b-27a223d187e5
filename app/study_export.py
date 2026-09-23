"""按匿名 StudyInstanceUID 下载检查子集。

复用已完成任务 output_dir 中的匿名文件（不重新解压/处理原始 ZIP）：

1. 以目标检查（匿名 StudyInstanceUID）的全部成功输出实例为初选集合；
2. 递归读取集合内匿名 DICOM 任意层级的 ReferencedSOPInstanceUID，把同任务内
   被引用的实例（可能属于其他检查）补入集合，直到引用闭包稳定；
3. 闭包不完整（引用目标不在成功输出中，或一个 SOP UID 对应多个输出文件）时
   返回全部冲突（匿名 UID + 原因），由调用方以 409 返回，不产出残缺 ZIP。
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field

import pydicom
from pydicom.sequence import Sequence

# 递归引用标签：与 tags.UID_KEYWORDS 中的实例引用语义一致，
# 这里只收集“指向某个 SOP 实例”的引用（任意序列深度）。
REFERENCED_SOP_KEYWORD = "ReferencedSOPInstanceUID"


@dataclass
class ReferenceConflict:
    code: str  # MISSING_REFERENCED_INSTANCE | AMBIGUOUS_REFERENCED_INSTANCE
    referenced_sop_instance_uid: str
    reason: str
    # AMBIGUOUS：同一匿名 SOP UID 对应的多个输出文件名
    output_names: list[str] = field(default_factory=list)
    # 哪些输出文件（扁平文件名）引用了该目标
    referenced_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "referenced_sop_instance_uid": self.referenced_sop_instance_uid,
            "reason": self.reason,
            "output_names": self.output_names,
            "referenced_by": self.referenced_by,
        }


@dataclass
class StudySubset:
    study_exists: bool
    primary_outputs: list[str] = field(default_factory=list)   # 初选（该检查全部实例）
    added_outputs: list[str] = field(default_factory=list)     # 闭包补入（含跨检查）
    selected_outputs: list[str] = field(default_factory=list)  # 最终全集（已排序）
    conflicts: list[ReferenceConflict] = field(default_factory=list)


def _eligible_reports(result) -> list:
    """成功输出（会进入匿名 ZIP）的逐文件报告，与 build_zip 的口径一致。"""
    return [
        r for r in result.files
        if r.status != "error" and r.output_name
    ]


def collect_referenced_sop_uids(path: str) -> set[str]:
    """读取一个匿名 DICOM，收集任意序列深度的 ReferencedSOPInstanceUID。

    匿名输出均为写出时校验过的合法 DICOM；停在像素数据之前即可，引用标签
    不会出现在像素之后。VM>1 的 UI 按反斜杠拆分为多个 UID。
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    found: set[str] = set()

    def _visit(current) -> None:
        stack = [current]
        while stack:
            node = stack.pop()
            for tag_ in list(node.keys()):
                elem = node[tag_]
                if elem.VR == "SQ" and isinstance(elem.value, Sequence):
                    stack.extend(elem.value)
                elif elem.keyword == REFERENCED_SOP_KEYWORD:
                    raw = elem.value
                    if raw is None:
                        continue
                    for part in str(raw).split("\\"):
                        part = part.strip()
                        if part:
                            found.add(part)

    _visit(ds)
    return found


def build_study_subset(job, study_instance_uid: str) -> StudySubset:
    """计算检查子集的引用闭包；不做任何 ZIP IO。"""
    result = job.result
    reports = _eligible_reports(result)

    primary = [r.output_name for r in reports
               if r.study_instance_uid == study_instance_uid]
    if not primary:
        return StudySubset(study_exists=False)

    # 匿名 SOP UID -> 成功输出文件名（正常 1 个；重复实例时可能多个）
    sop_to_outputs: dict[str, list[str]] = {}
    for r in reports:
        if r.sop_instance_uid:
            sop_to_outputs.setdefault(r.sop_instance_uid, []).append(r.output_name)
    for outs in sop_to_outputs.values():
        outs.sort()

    selected: set[str] = set(primary)
    scanned: set[str] = set()
    queue: list[str] = list(primary)
    conflicts: dict[tuple[str, str], ReferenceConflict] = {}

    def _note_conflict(code: str, uid: str, output_names: list[str],
                       referenced_by: str) -> None:
        key = (code, uid)
        conflict = conflicts.get(key)
        if conflict is None:
            if code == "MISSING_REFERENCED_INSTANCE":
                reason = (
                    "被引用的 SOP 实例不在本任务的成功匿名输出中"
                    "（原始处理失败/缺失或为包外悬空引用），无法保证引用完整"
                )
            else:
                reason = (
                    "一个匿名 SOPInstanceUID 对应多个输出文件（原始数据存在重复 "
                    "SOP 实例），无法确定应补入哪一个"
                )
            conflict = ReferenceConflict(
                code=code,
                referenced_sop_instance_uid=uid,
                reason=reason,
                output_names=output_names,
            )
            conflicts[key] = conflict
        if referenced_by not in conflict.referenced_by:
            conflict.referenced_by.append(referenced_by)

    while queue:
        output_name = queue.pop()
        if output_name in scanned:
            continue
        scanned.add(output_name)
        targets = collect_referenced_sop_uids(
            os.path.join(result.output_dir, output_name)
        )
        for target in targets:
            outs = sop_to_outputs.get(target, [])
            if len(outs) == 0:
                _note_conflict("MISSING_REFERENCED_INSTANCE", target, [],
                               output_name)
            elif len(outs) > 1:
                _note_conflict("AMBIGUOUS_REFERENCED_INSTANCE", target, outs,
                               output_name)
            else:
                resolved = outs[0]
                if resolved not in selected:
                    selected.add(resolved)
                    queue.append(resolved)

    ordered_conflicts = [
        conflicts[key] for key in sorted(conflicts, key=lambda k: (k[0], k[1]))
    ]
    primary_set = set(primary)
    return StudySubset(
        study_exists=True,
        primary_outputs=sorted(primary_set),
        added_outputs=sorted(selected - primary_set),
        selected_outputs=sorted(selected),
        conflicts=ordered_conflicts,
    )


def build_subset_zip(output_dir: str, output_names: list[str],
                     dest_zip: str) -> str:
    """将选定的匿名输出以扁平结构打包（arcname 即扁平输出文件名）。"""
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for output_name in sorted(output_names):
            src = os.path.join(output_dir, output_name)
            zf.write(src, arcname=output_name)
    return dest_zip
