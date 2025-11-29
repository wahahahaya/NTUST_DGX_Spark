"""
title: YOLO Object Detector
author: Arlen
description: 將聊天中的圖片送到外部 YOLO FastAPI 伺服器 ( /predict ) 做物件偵測。
required_open_webui_version: 0.6.36
version: 0.1.0
requirements: httpx
license: MIT
"""

import os
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any

import httpx
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        YOLO_SERVER_URL: str = Field(
            default="http://127.0.0.1:6611",
            description="YOLO FastAPI 伺服器的 base URL，例如 http://127.0.0.1:6611",
        )
        UPLOADS_DIR: str = Field(
            default="/home/ntust_spark/playbook_gptoss/webui/uploads",
            description="Open WebUI 上傳檔案實際所在的 uploads 目錄（絕對路徑）",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ---- 工具主函式 ----
    async def detect_with_yolo(
        self,
        __files__: Optional[List[Dict[str, Any]]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        從目前對話中的圖片取得影像，送給 YOLO FastAPI 的 /predict，並回傳偵測結果。

        優先使用 __files__（使用者上傳的檔案），
        若無，則從 __messages__ 中尋找 data:image/...;base64,... 格式的 image_url。
        """

        # 1. 先嘗試從 __files__ 取得檔案路徑（前提：使用者用「Upload Files」上傳）:contentReference[oaicite:3]{index=3}
        image_bytes: Optional[bytes] = None
        meta: Dict[str, Any] = {}

        if __files__:
            # 只取第一個檔案；你可以依需求改成處理多張
            f0 = __files__[0]

            files_info = f0.get("files") or {}
            file_id = files_info.get("id")
            filename = files_info.get("filename") or "image"

            if file_id:
                uploads_dir = self.valves.UPLOADS_DIR
                if not os.path.isabs(uploads_dir):
                    raise ValueError(
                        f"UPLOADS_DIR 必須是絕對路徑，目前為: {uploads_dir}"
                    )

                # 根據官方文件，實際路徑格式為 <UPLOADS_DIR>/<id>_<filename>:contentReference[oaicite:4]{index=4}
                file_path = os.path.join(uploads_dir, f"{file_id}_{filename}")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"在 UPLOADS_DIR 中找不到檔案: {file_path}（請確認 UPLOADS_DIR 設定是否正確）"
                    )

                with open(file_path, "rb") as f:
                    image_bytes = f.read()

                meta["source"] = "file"
                meta["file_path"] = file_path

        # 2. 如果 __files__ 沒有 usable image，就從 __messages__ 抓 data:image;base64
        if image_bytes is None and __messages__:
            # 從最後一則 user 訊息往前掃
            for msg in reversed(__messages__):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue

                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            # data:image/png;base64,xxxx...
                            try:
                                header, b64data = url.split(",", 1)
                                image_bytes = base64.b64decode(b64data)
                                meta["source"] = "data_url"
                                meta["mime_header"] = header
                                break
                            except Exception as e:
                                meta["data_url_error"] = str(e)
                if image_bytes is not None:
                    break

        if image_bytes is None:
            return {
                "error": "在目前的對話中沒有找到可用的圖片（既沒有檔案也沒有 data:image;base64）。",
                "hint": "請用『Upload Files』按鈕上傳圖片，或直接貼一張圖片，再要求模型呼叫 detect_with_yolo 工具。",
                "meta": meta,
            }

        # 3. 呼叫 YOLO FastAPI 的 /predict
        yolo_url = self.valves.YOLO_SERVER_URL.rstrip("/") + "/predict"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                files = {
                    "file": ("image.jpg", image_bytes, "image/jpeg"),
                }
                resp = await client.post(yolo_url, files=files)
                resp.raise_for_status()
                data = resp.json()

                annotated_url = data.get("annotated_image_url")
        except httpx.HTTPError as e:
            return {
                "error": f"呼叫 YOLO 服務失敗: {e}",
                "yolo_url": yolo_url,
                "meta": meta,
            }
        except Exception as e:
            return {
                "error": f"工具內部錯誤: {e}",
                "yolo_url": yolo_url,
                "meta": meta,
            }
        markdown = None
        if annotated_url:
            markdown = "偵測結果如下：\n\n" f"![YOLO 偵測結果]({annotated_url})"
        # 4. 把 YOLO 回傳結果 + 一些來源資訊一併回傳
        return {
            "detections": data.get("detections", []),
            "annotated_image_url": annotated_url,
            "markdown": markdown,
            "meta": meta,
            "message": "YOLO 物件偵測完成，請在回覆中嵌入 markdown 顯示圖片。",
        }
