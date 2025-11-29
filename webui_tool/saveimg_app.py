"""
title: Save Chat Images
author: Arlen
description: 將目前對話中附加的圖片檔案，解碼後儲存到伺服器指定目錄。
required_open_webui_version: 0.6.36
version: 0.1.0
requirements:
license: MIT
"""

import os
import base64
import json
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        OUTPUT_DIR: str = Field(
            default="/home/ntust_spark/playbook_gptoss/webui/chat_images",
            description="圖片輸出目錄（請務必填絕對路徑，且需有寫入權限）",
        )

    def __init__(self):
        # 管理員在 UI 裡修改 Valves 後，這裡就會拿到新的設定值
        self.valves = self.Valves()

    async def save_chat_images(
        self,
        __files__: Optional[List[Dict[str, Any]]] = None,
        __messages__: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        將目前聊天中附加的圖片，儲存成檔案到 OUTPUT_DIR 目錄下。

        :param __files__: 由 Open WebUI 注入的「目前訊息附帶檔案」列表。
        :param __messages__: 由 Open WebUI 注入的「整段對話訊息」列表，用來額外掃描 image_url。
        :return: JSON 字串，描述成功與略過的項目。
        """

        output_dir = self.valves.OUTPUT_DIR

        # 強制要求 OUTPUT_DIR 是絕對路徑，避免被 LLM 帶歪
        if not os.path.isabs(output_dir):
            raise ValueError(f"OUTPUT_DIR 必須是絕對路徑，目前為: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        image_entries: List[Dict[str, Any]] = []

        # 1) 先從 __files__ 抓（這是官方文件說的「附加檔案」）:contentReference[oaicite:2]{index=2}
        if __files__:
            for f in __files__:
                if f.get("type") == "image":
                    image_entries.append(f)

        # 2) 再從 __messages__ 裡面抓 OpenAI 樣式的 image_url（參考 Auto Image Tool 寫法）:contentReference[oaicite:3]{index=3}
        if __messages__:
            for msg in reversed(__messages__):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url")
                            if url:
                                image_entries.append({"type": "image", "url": url})

        if not image_entries:
            return json.dumps(
                {
                    "saved": [],
                    "skipped": [],
                    "message": "目前對話中沒有找到任何圖片附件。",
                },
                ensure_ascii=False,
            )

        saved_paths: List[str] = []
        skipped: List[Dict[str, Any]] = []

        for idx, entry in enumerate(image_entries):
            url = entry.get("url")
            original_name = entry.get("name") or f"chat_image_{idx}"

            if not url:
                skipped.append(
                    {"index": idx, "name": original_name, "reason": "缺少 url 欄位"}
                )
                continue

            # 情境一：data:image/...;base64,xxxx 這類 inline base64 URL
            if url.startswith("data:image"):
                try:
                    header, b64data = url.split(",", 1)
                except ValueError:
                    skipped.append(
                        {
                            "index": idx,
                            "name": original_name,
                            "reason": "data URL 格式錯誤",
                        }
                    )
                    continue

                # 從 mime type 抓副檔名，失敗就給個預設
                ext = "bin"
                try:
                    mime_part = header.split(";")[0]  # e.g. data:image/png
                    ext = mime_part.split("/")[1]
                except Exception:
                    pass

                try:
                    raw = base64.b64decode(b64data)
                except Exception as e:
                    skipped.append(
                        {
                            "index": idx,
                            "name": original_name,
                            "reason": f"base64 解碼失敗: {e}",
                        }
                    )
                    continue

                # 簡單做一下檔名清理
                safe_name = "".join(c for c in original_name if c not in '\\/:*?"<>|')
                if not safe_name:
                    safe_name = f"chat_image_{idx}"
                filename = f"{safe_name}.{ext}"
                full_path = os.path.join(output_dir, filename)

                # 避免覆寫舊檔
                base, extension = os.path.splitext(full_path)
                counter = 1
                while os.path.exists(full_path):
                    full_path = f"{base}_{counter}{extension}"
                    counter += 1

                with open(full_path, "wb") as f:
                    f.write(raw)

                saved_paths.append(full_path)

            else:
                # 非 data: URL（例如 /api/v1/files/... 或 http://...）：
                # 在這個 Tool 先不幫你亂猜主機與路徑，避免誤打到外網或錯 host。
                skipped.append(
                    {
                        "index": idx,
                        "name": original_name,
                        "url": url,
                        "reason": "目前 Tool 僅支援 data:image;base64 形式的圖片 URL",
                    }
                )

        result = {
            "saved": saved_paths,
            "skipped": skipped,
            "message": f"已儲存 {len(saved_paths)} 個檔案，略過 {len(skipped)} 個。",
        }
        return json.dumps(result, ensure_ascii=False)
