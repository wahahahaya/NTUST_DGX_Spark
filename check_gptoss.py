from openai import OpenAI

BASE_URL = "http://localhost:8355/v1"
MODEL_NAME = "openai/gpt-oss-20b"  # 要跟你 trtllm-serve 的 model 名稱一致

def main():
    client = OpenAI(
        base_url=BASE_URL,
        api_key="dummy-key"  # TRT-LLM 通常不真的驗證 key
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "請簡短介紹一下你目前是跑在 TensorRT-LLM 上的 GPT-OSS。"}
            ],
            max_tokens=64,
        )

        msg = resp.choices[0].message
        print("=== GPT-OSS 回應 ===")
        print(msg.content)

    except Exception as e:
        print("呼叫失敗，請檢查：")
        print(f"- BASE_URL：{BASE_URL}")
        print(f"- MODEL_NAME：{MODEL_NAME}")
        print("Exception：", repr(e))


if __name__ == "__main__":
    main()
